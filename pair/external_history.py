import json
from typing import Union
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import requests
from pair.constants import CAPITAL_TIMEFRAME
from tenacity import retry, stop_after_attempt


def get_yahoo_data(symbol: str, interval="1m", n_periods: int = 3, max_rows: int = None) -> Union[pd.DataFrame, None]:
    stock = yf.Ticker(symbol)
    periods = {"1m": {"n_periods": n_periods, "period": "7d", "step_days": 7},
               "5m": {"n_periods": 1, "period": "1mo", "step_days": 30},
               "15m": {"n_periods": 1, "period": "1mo", "step_days": 30},
               "30m": {"n_periods": 1, "period": "1mo", "step_days": 30},
               "1h": {"n_periods": 1, "period": "1mo", "step_days": 30},
               "1d": {"n_periods": 1, "period": "1mo", "step_days": 30}}
    try:

        history = stock.history(period=periods[interval]["period"], interval=interval)

        for i in range(1, periods[interval]["n_periods"]):
            earliest_time = history.index.min()
            earliest_time = datetime(earliest_time.year, earliest_time.month, earliest_time.day,
                                     earliest_time.hour, earliest_time.minute)

            start = earliest_time - timedelta(days=periods[interval]["step_days"]) + timedelta(hours=1)
            history_early = stock.history(interval=interval,
                                          start=start,
                                          end=earliest_time)

            history_early.drop(history_early.index.intersection(history.index), inplace=True)
            history = pd.concat([history_early, history], axis=0)

        history.drop(["Volume", "Dividends", "Stock Splits"], axis=1, inplace=True)
        history.set_axis(["open", "high", "low", "close"], axis=1, inplace=True)

        history.sort_index(axis=0, inplace=True)

        if interval == "1h":  # delete the last partial period
            if (history.index[-1] - history.index[-2]).seconds < 3600:
                history.drop(history.index[-1], inplace=True)

        if max_rows is not None:
            history = history[-max_rows:]
        return history

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
    def get_capital_data(self, ds_symbol: str, resolution: int, numpoints: int = 100) -> Union[list, None]:

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


if __name__ == "__main__":
    api_key, identifier, password = json.load(open("../multi_config.json"))["capital_creds"].values()
    conn = CapitalConnection(api_key=api_key, identifier=identifier, password=password)
    data = conn.get_capital_data("EURGBP", resolution=5, numpoints=500)
    print(data)
    # data.to_csv("Tests/SP500_yahoo_data.csv")
