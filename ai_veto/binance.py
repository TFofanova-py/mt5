import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import List
import asyncio
import time
import hmac
import hashlib
import os
from dotenv import load_dotenv
import logging
import traceback
from jinja2 import Environment, FileSystemLoader
import pandas as pd

logger = logging.getLogger(__name__)

load_dotenv()

env = Environment(loader=FileSystemLoader('.'))


class Binance:
    def __init__(self):
        self.info_url = "https://api.binance.com/api/v3/exchangeInfo"
        self.depth_url = "https://api.binance.com/api/v3/depth"
        self.network_url = "https://api.binance.com/sapi/v1/capital/config/getall"
        self.futures_info_url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
        self.funding_rate_url = "https://fapi.binance.com/fapi/v1/fundingRate"
        self.open_interest_url = "https://fapi.binance.com/futures/data/openInterestHist"
        self.global_long_short_ratio_url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
        self.taker_flow_url = "https://fapi.binance.com/futures/data/takerlongshortRatio"
        self.api_key = os.getenv(f"BINANCE_API_KEY")
        self.secret_key = os.getenv(f"BINANCE_SECRET_KEY")


    @retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential())
    async def get_depth(self, pair: str, limit: int = 1) -> dict:
        url = self.depth_url
        params = {"symbol": pair, "limit": limit}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                try:
                    data = await response.json()
                    data["bids"] = [{"price": float(tpl[0]), "volume": float(tpl[1])} for tpl in data["bids"]]
                    data["asks"] = [{"price": float(tpl[0]), "volume": float(tpl[1])} for tpl in data["asks"]]
                    return data


                except (aiohttp.client_exceptions.ContentTypeError, KeyError, TypeError):
                    print(f"Error in get_depth, bad response, pair={pair}, response={response}")
                    raise asyncio.TimeoutError

                except Exception as e:
                    print(f"Error in get_depth, pair={pair}")



    async def get_headers_and_params(self, params: dict = None):
        headers = {
            "X-MBX-APIKEY": self.api_key
        }

        timestamp = int(time.time() * 1000)
        query_string = f"timestamp={timestamp}"
        signature = hmac.new(self.secret_key.encode(), query_string.encode(), hashlib.sha256).hexdigest()

        extra_params = {"timestamp": timestamp, "signature": signature}
        if params:
            params.update(extra_params)
        else:
            params = extra_params
        return headers, params


    async def health_check(self) -> str:
        headers, params = await self.get_headers_and_params()
        url = self.network_url
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                data = await response.json()
                if isinstance(data, dict):
                    return f"Binance, health check: {data}"
                return f"Binance, health check: OK"


    async def get_funding_rate(self, pair: str, limit: int = 5):
        headers, params = await self.get_headers_and_params(params={"symbol": pair, "limit": limit})
        url = self.funding_rate_url
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                data =  await response.json()
                df = pd.DataFrame(data)
                df["fundingRate"] = df["fundingRate"].astype(float)
                df["zscore"] = (df["fundingRate"] - df["fundingRate"].mean()) / df["fundingRate"].std()
                return df.iloc[-1].to_dict()


    async def get_open_interest(self, pair: str, period: str = "5m", limit: int = 5):
        headers, params = await self.get_headers_and_params(params={"symbol": pair, "period": period, "limit": limit})
        url = self.open_interest_url
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                data =  await response.json()
                return [{k: x.get(k) for k in ["timestamp", "sumOpenInterest", "sumOpenInterestValue", "CMCCirculatingSupply"]} for x in data]


    async def get_global_long_short_ratio(self, pair: str, period: str = "5m", limit: int = 5):
        headers, params = await self.get_headers_and_params(params={"symbol": pair, "period": period, "limit": limit})
        url = self.global_long_short_ratio_url
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                data =  await response.json()
                return [{k: x.get(k) for k in ["timestamp", "longShortRatio"]} for x in data]


    async def get_taker_flow(self, pair: str, period: str = "5m", limit: int = 5):
        headers, params = await self.get_headers_and_params(params={"symbol": pair, "period": period, "limit": limit})
        url = self.taker_flow_url
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                data = await response.json()
                return [{k: x.get(k) for k in ["timestamp", "buySellRatio"]} for x in data]

    async def get_current_data(self, pair: str):
        order_book = await self.get_depth(pair=pair, limit=5)
        funding_rate = await self.get_funding_rate(pair=pair, limit=100)
        open_interest = await self.get_open_interest(pair=pair, limit=5)
        global_long_short_ratio = await self.get_global_long_short_ratio(pair=pair, limit=5)
        taker_flow = await self.get_taker_flow(pair=pair, limit=5)

        return order_book, funding_rate, open_interest, global_long_short_ratio, taker_flow

    async def get_prompt(self, coin: str, direction: str, divergences: List[str] = None, ohlc_data: List[dict] = None) -> str:
        pair = f"{coin}USDT"
        order_book, funding_rate, open_interest, global_long_short_ratio, taker_flow = await self.get_current_data(pair=pair)
        template = env.get_template("ai_veto/prompt.jinja2")
        return template.render(coin=coin,
                                 direction=direction,
                                 divergences=divergences,
                                 ohlc_data=ohlc_data,
                                 timestamp=int(time.time() * 1000),
                                 order_book=order_book,
                                 funding_rate=funding_rate,
                                 open_interest=open_interest,
                                 global_long_short_ratio=global_long_short_ratio,
                                 taker_flow=taker_flow)



if __name__ == "__main__":
    binance = Binance()
    ohlc = [{'close': 3086.02, 'open': 3091.62, 'high': 3094.87, 'low': 3085.17, 'volume': 157, 'timestamp': '2025-11-16 20:06:00+00:00'}, {'close': 3086.67, 'open': 3086.02, 'high': 3090.27, 'low': 3081.07, 'volume': 165, 'timestamp': '2025-11-16 20:07:00+00:00'}, {'close': 3084.42, 'open': 3087.72, 'high': 3089.02, 'low': 3081.12, 'volume': 143, 'timestamp': '2025-11-16 20:08:00+00:00'}, {'close': 3093.77, 'open': 3084.27, 'high': 3096.62, 'low': 3084.27, 'volume': 158, 'timestamp': '2025-11-16 20:09:00+00:00'}, {'close': 3099.07, 'open': 3093.77, 'high': 3100.92, 'low': 3093.77, 'volume': 156, 'timestamp': '2025-11-16 20:10:00+00:00'}, {'close': 3119.27, 'open': 3098.82, 'high': 3119.27, 'low': 3098.32, 'volume': 156, 'timestamp': '2025-11-16 20:11:00+00:00'}, {'close': 3114.47, 'open': 3117.92, 'high': 3121.57, 'low': 3107.32, 'volume': 204, 'timestamp': '2025-11-16 20:12:00+00:00'}, {'close': 3111.77, 'open': 3114.62, 'high': 3117.37, 'low': 3107.57, 'volume': 187, 'timestamp': '2025-11-16 20:13:00+00:00'}, {'close': 3111.22, 'open': 3111.07, 'high': 3113.82, 'low': 3108.27, 'volume': 176, 'timestamp': '2025-11-16 20:14:00+00:00'}, {'close': 3113.62, 'open': 3111.77, 'high': 3113.97, 'low': 3111.62, 'volume': 9, 'timestamp': '2025-11-16 20:15:00+00:00'}]
    print(asyncio.run(binance.get_prompt(coin="BTC", direction="long", ohlc_data=ohlc, divergences=["BOO", "FOO"])))
