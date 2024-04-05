import json
from typing import Dict, Optional
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import requests
from pair.constants import CAPITAL_TIMEFRAME
from tenacity import retry, stop_after_attempt


def post_processing_yahoo(df: pd.DataFrame, numpoints: int) -> pd.DataFrame:
    df.drop(["Dividends", "Stock Splits"], axis=1, inplace=True, errors="ignore")
    df.rename({"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"},
                   axis=1, inplace=True)
    df.drop_duplicates(inplace=True)
    df.sort_index(axis=0, inplace=True)
    return df.iloc[-numpoints:] if df.shape[0] >= numpoints else df


def get_yahoo_data(symbol: str, interval="1m", numpoints: int = 100) -> Optional[pd.DataFrame]:
    stock = yf.Ticker(symbol)

    try:
        history = stock.history(interval=interval)

        if history.shape[0] > 0:
            return post_processing_yahoo(history, numpoints=numpoints)

    except Exception as e:
        print(e)

    return None


class CapitalConnection():
    def __init__(self, api_key: str, identifier: str, password: str):
        self.headers = None

        req_headers = {'X-CAP-API-KEY': api_key, 'Content-Type': 'application/json'}
        payload = {"encryptedPassword": "false", "identifier": identifier, "password": password}

        res = requests.post("https://api-capital.backend-capital.com/api/v1/session",
                            json=payload,
                            headers=req_headers)
        if res.status_code == 200:
            self.headers = {"CST": res.headers["CST"],
                            "X-SECURITY-TOKEN": res.headers["X-SECURITY-TOKEN"]}

    @retry(stop=stop_after_attempt(5))
    def get_capital_data(self, ds_symbol: str, resolution: int, numpoints: int = 100) -> Optional[list]:

        capital_resolution = CAPITAL_TIMEFRAME[resolution]
        url = "https://api-capital.backend-capital.com/api/v1/prices/" + ds_symbol + "?resolution=" + capital_resolution + "&max=" + str(
            numpoints)
        res = requests.get(url, headers=self.headers)

        if res.status_code == 200:
            prices = json.loads(res.text)["prices"]

            return prices

        print("Error:", ds_symbol, res.content)
        return None

    @retry(stop=stop_after_attempt(5))
    def search_for_epic(self, search_term: str):
        url = "https://api-capital.backend-capital.com/api/v1/markets?searchTerm=" + search_term
        res = requests.get(url, headers=self.headers)

        if res.status_code == 200:
            return json.loads(res.text)

        return None


def get_twelvedata(symbol: str, resolution: int, numpoints: int = 100) -> pd.DataFrame:
    tw_api_key = '8f29dcf98c484ec98d98c2c3127af636'
    url = 'https://api.twelvedata.com/time_series'
    interval = None
    if resolution < 60:
        interval = str(resolution) + 'min'
    elif resolution < 24 * 60:
        interval = str(resolution // 60) + 'h'
    elif resolution == 24 * 60:
        interval = '1day'
    elif resolution == 7 * 24 * 60:
        interval = '1week'

    assert interval in ['1min', '5min', '15min', '30min', '45min', '1h', '2h', '4h', '8h', '1day', '1week'], \
        'Invalid resolution for twelevedata.com'
    params = {
        'symbol': symbol,
        'interval': interval,
        'outputsize': str(numpoints),
        'format': 'JSON',
        'apikey': tw_api_key
    }

    response = requests.get(url, params=params)

    if response.ok:
        data = response.json()
        df = pd.DataFrame(data['values'])
        df.index = pd.to_datetime(df['datetime'])
        df = df.drop('datetime', axis=1)

        for col in df.columns:
            df[col] = df[col].astype(float)

        return df
    else:
        print("Error:", response.status_code)


if __name__ == "__main__":
    # api_key, identifier, password = json.load(open("../multi_config.json"))["capital_creds"].values()
    # conn = CapitalConnection(api_key=api_key, identifier=identifier, password=password)
    # data = conn.get_capital_data("EURGBP", resolution=5, numpoints=500)
    data = get_twelvedata(symbol='USD/EUR', resolution=60)
    print(data)
    # data.to_csv("Tests/SP500_yahoo_data.csv")
