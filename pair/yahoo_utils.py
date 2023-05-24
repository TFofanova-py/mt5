from typing import Union
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd


def get_yahoo_data(symbol: str, interval="1m") -> Union[pd.DataFrame, None]:
    stock = yf.Ticker(symbol)
    periods = {"1m": {"n_periods": 3, "period": "7d", "step_days": 7},
               "1h": {"n_periods": 12, "period": "1mo", "step_days": 30}}
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

        return history

    except Exception as e:
        print(e)

    return None


if __name__ == "__main__":
    data = get_yahoo_data("ES=F")
    data.to_csv("Tests/SP500_yahoo_data.csv")
