from typing import Union
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd


def get_yahoo_data(symbol: str) -> Union[pd.DataFrame, None]:
    stock = yf.Ticker(symbol)
    try:
        n_weeks = 3
        history = stock.history(period="7d", interval="1m")

        for i in range(1, n_weeks):
            earliest_time = history.index.min()
            earliest_time = datetime(earliest_time.year, earliest_time.month, earliest_time.day,
                                     earliest_time.hour, earliest_time.minute)
            start = earliest_time - timedelta(days=7) + timedelta(hours=1)
            history_early = stock.history(interval="1m",
                                          start=start,
                                          end=earliest_time)

            history_early.drop(history_early.index.intersection(history.index), inplace=True)
            history = pd.concat([history_early, history], axis=0)

        history.drop(["Volume", "Dividends", "Stock Splits"], axis=1, inplace=True)
        history.set_axis(["open", "high", "low", "close"], axis=1, inplace=True)

        history.sort_index(axis=0, inplace=True)
        return history

    except Exception as e:
        print(e)

    return None


if __name__ == "__main__":
    data = get_yahoo_data("ES=F")
    data.to_csv("Tests/SP500_yahoo_data.csv")
