import pandas as pd
import numpy as np
from typing import Literal


def crossover(ts_1: pd.Series, ts_2: pd.Series) -> bool:  # ts_1 crossovers ts_2
    if (ts_1.iloc[-2] < ts_2.iloc[-2]) and (ts_1.iloc[-1] > ts_2.iloc[-1]):
        return True
    return False


def crossunder(ts_1: pd.Series, ts_2: pd.Series) -> bool:  # ts_1 crossunders ts_2
    if (ts_1.iloc[-2] > ts_2.iloc[-2]) and (ts_1.iloc[-1] < ts_2.iloc[-1]):
        return True
    return False


def pivotlevel(ar: np.array, right_bars: int, level_type: Literal["low", "high"]) -> float:
    extremum_func = np.argmin if level_type == "low" else np.argmax
    extremum_to_end_dist = ar.shape[0] - 1 - extremum_func(ar[::-1])
    if extremum_to_end_dist == right_bars:
        idx = ar.shape[0] - extremum_to_end_dist - 1
        return ar[idx]
    return np.nan


def donchian(df: pd.DataFrame, period: int) -> pd.Series:
    assert all([x in df.columns for x in ["low", "high"]]), "DataFrame must have 'low' and 'high' as columns"
    return (df["low"].rolling(window=period, min_periods=period).min() + df["high"].rolling(window=period, min_periods=period).max()) / 2


def rsi(df: pd.DataFrame, period: int = 14, upper_threshold: int = 70, lower_threshold: int = 30) -> pd.DataFrame:
    rsi_df = df.copy()
    rsi_df["close_1"] = rsi_df["close"].shift(1)
    rsi_df["yield"] = (rsi_df["close"] - rsi_df["close_1"]) / rsi_df["close_1"] * 100
    rsi_df["u"] = rsi_df["yield"].apply(lambda x: max(0, x))
    rsi_df["d"] = rsi_df["yield"].apply(lambda x: -min(0, x))
    rsi_df["avg_u"] = rsi_df["u"].rolling(window=period, min_periods=period).mean()
    rsi_df["avg_d"] = rsi_df["d"].rolling(window=period, min_periods=period).mean()
    rsi_df["rsi"] = 100 - (100 / (1 + rsi_df["avg_u"] / rsi_df["avg_d"]))
    rsi_df["rsi_upper_threshold"] = upper_threshold
    rsi_df["rsi_lower_threshold"] = lower_threshold
    return rsi_df


def bollinger_bands(bb_df: pd.DataFrame, period: int = 200, mult: int = 2) -> pd.DataFrame:
    bb_df["basis_" + str(mult)] = bb_df["close"].rolling(window=period, min_periods=period).mean()
    bb_df["dev_" + str(mult)] = mult * bb_df["close"].rolling(window=period, min_periods=period).std(ddof=0)
    bb_df["upper_" + str(mult)] = bb_df["basis_" + str(mult)] + bb_df["dev_" + str(mult)]
    bb_df["lower_" + str(mult)] = bb_df["basis_" + str(mult)] - bb_df["dev_" + str(mult)]
    return bb_df


def sharp_n_true(ser: pd.Series, n: int) -> bool:
    if ser.shape[0] < n:
        return False
    elif ser.shape[0] == n and all(ser):
        return True
    return all(ser.iloc[-n:]) and not ser.iloc[-n - 1]
