import pandas as pd


def crossover(ts_1: pd.Series, ts_2: pd.Series) -> bool:  # ts_1 crossovers ts_2
    if (ts_1.iloc[-2] < ts_2.iloc[-2]) and (ts_1.iloc[-1] > ts_2.iloc[-1]):
        return True
    return False


def crossunder(ts_1: pd.Series, ts_2: pd.Series) -> bool:  # ts_1 crossunders ts_2
    if (ts_1.iloc[-2] > ts_2.iloc[-2]) and (ts_1.iloc[-1] < ts_2.iloc[-1]):
        return True
    return False