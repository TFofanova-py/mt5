import pandas as pd
import numpy as np
from typing import Tuple, Union
from pair.enums import IchimokuTrend


def crossover(ts_1: pd.Series, ts_2: pd.Series) -> bool:  # ts_1 crossovers ts_2
    if (ts_1.iloc[-2] < ts_2.iloc[-2]) and (ts_1.iloc[-1] > ts_2.iloc[-1]):
        return True
    return False


def crossunder(ts_1: pd.Series, ts_2: pd.Series) -> bool:  # ts_1 crossunders ts_2
    if (ts_1.iloc[-2] > ts_2.iloc[-2]) and (ts_1.iloc[-1] < ts_2.iloc[-1]):
        return True
    return False


def pivotlow(ser: pd.Series, left_bars: int, right_bars: int = None) -> Tuple[pd.Series, pd.Series]:

    ser = ser.reset_index(drop=True)

    if right_bars is None:
        right_bars = left_bars

    range_bars = left_bars + right_bars + 1
    argmin_ser = (
        ser
            .rolling(window=range_bars, min_periods=range_bars)
            .apply(lambda w: np.argmin(w[::-1]))
            .dropna()
            .astype(int)
    )

    pl_vals = pd.Series([np.nan, ] * len(ser), dtype=float).iloc[argmin_ser.index]
    pl_positions = pd.Series([np.nan, ] * len(ser), dtype=float).iloc[argmin_ser.index]
    low_indices = pl_vals[argmin_ser == right_bars].index
    pl_vals.loc[low_indices] = ser.shift(right_bars).loc[low_indices]
    pl_positions.loc[low_indices] = low_indices.values
    # pl_vals = pl_vals.fillna(method="ffill").dropna()
    # pl_positions = pl_positions.fillna(method="ffill").dropna().astype(int)
    pl_vals = pl_vals.dropna()
    pl_positions = pl_positions.dropna().astype(int)

    return pl_vals, pl_positions


def pivothigh(ser: pd.Series, left_bars: int, right_bars: int = None) -> Tuple[pd.Series, pd.Series]:

    ser = ser.reset_index(drop=True)

    if right_bars is None:
        right_bars = left_bars

    range_bars = left_bars + right_bars + 1
    argmax_ser = (
        ser
            .rolling(window=range_bars, min_periods=range_bars)
            .apply(lambda w: np.argmax(w[::-1]))
            .dropna()
            .astype(int)
    )

    ph_vals = pd.Series([np.nan, ] * len(ser), dtype=float).iloc[argmax_ser.index]
    ph_positions = pd.Series([np.nan, ] * len(ser), dtype=float).iloc[argmax_ser.index]
    high_indices = ph_vals[argmax_ser == right_bars].index
    ph_vals.loc[high_indices] = ser.shift(right_bars).loc[high_indices]
    ph_positions.loc[high_indices] = high_indices.values
    # ph_vals = ph_vals.fillna(method="ffill").dropna()
    # ph_positions = ph_positions.fillna(method="ffill").dropna().astype(int)
    ph_vals = ph_vals.dropna()
    ph_positions = ph_positions.dropna().astype(int)

    return ph_vals, ph_positions


def donchian(df: pd.DataFrame, period: int) -> pd.Series:
    assert all([x in df.columns for x in ["low", "high"]]), "DataFrame must have 'low' and 'high' as columns"
    return (df["low"].rolling(window=period, min_periods=period).min() + df["high"].rolling(window=period, min_periods=period).max()) / 2


def rsi(df: pd.DataFrame, period: int = 14, upper_threshold: int = 70, lower_threshold: int = 30) -> pd.DataFrame:

    # Calculate Relative Strength Index - RSI

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


def bollinger_bands(df: pd.DataFrame, period: int = 200, mult: float = 2.0) -> pd.DataFrame:
    df["basis"] = df["close"].rolling(window=period, min_periods=1).mean()
    df["dev"] = mult * df["close"].rolling(window=period, min_periods=1).std(ddof=0)
    df["upper"] = df["basis"] + df["dev"]
    df["lower"] = df["basis"] - df["dev"]
    return df


def macd(df: pd.DataFrame, fast_length: int = 12, slow_length: int = 26, signal_length: int = 9) -> pd.DataFrame:

    # Calculate Moving Average Convergence Divergence - MACD, signal, deltamacd
    df_macd = df.copy()
    fast_ma = df_macd["close"].rolling(window=fast_length, min_periods=fast_length).mean()
    slow_ma = df_macd["close"].rolling(window=slow_length, min_periods=slow_length).mean()
    df_macd["macd"] = fast_ma - slow_ma
    df_macd["signal"] = df_macd["macd"].rolling(window=signal_length, min_periods=signal_length).mean()
    df_macd["hist"] = df_macd["macd"] - df_macd["signal"]  # deltamacd?
    return df_macd


def momentum(df: pd.DataFrame, length: int = 10) -> pd.DataFrame:

    # Calculate Momentum
    df_mom = df.copy()
    df_mom["momentum"] = df["close"] - df["close"].shift(length)
    return df_mom


def cci(df: pd.DataFrame, length: int = 20) -> pd.DataFrame:

    # Calculate Commodity Channel Index - CCI

    df_cci = df.copy()
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3
    ma = hlc3.rolling(window=length, min_periods=length).mean()
    dev = abs(hlc3 - ma).rolling(window=length, min_periods=length).mean()
    df_cci["cci"] = (hlc3 - ma) / (0.015 * dev)
    return df_cci


def obv(df: pd.DataFrame) -> pd.DataFrame:

    # Calculate On Balance Volume - OBV

    df_obv = df.copy()
    sign = ((df["close"] - df["close"].shift(1)) >= 0).astype(int)
    df_obv["obv"] = (sign * df["volume"]).cumsum()

    return df_obv


def stk(df: pd.DataFrame, stoch_length: int = 14, sma_length: int = 3) -> pd.DataFrame:

    # Calculate Stochastic

    df_stk = df.copy()
    lowest = df["low"].rolling(window=stoch_length, min_periods=stoch_length).min()
    highest = df["high"].rolling(window=stoch_length, min_periods=stoch_length).max()
    stoch = 100 * (df["close"] - lowest) / (highest - lowest)
    df_stk["stk"] = stoch.rolling(window=sma_length, min_periods=sma_length).mean()

    return df_stk


def vwmacd(df: pd.DataFrame, fast_length: int = 12, slow_length: int = 26) -> pd.DataFrame:

    # Calculate Volume Weighted Moving Average Convergence Divergence - VWMACD

    def vwma(data: pd.DataFrame, length: int = 1) -> pd.Series:
        ma_vol_close = (data["close"] * data["volume"]).rolling(window=length, min_periods=length).mean()
        ma_vol = data["volume"].rolling(window=length, min_periods=length).mean()
        return ma_vol_close / ma_vol

    df_vw = df.copy()
    vwma_fast = vwma(df, fast_length)
    vwma_slow = vwma(df, slow_length)
    df_vw["vwmacd"] = vwma_fast - vwma_slow
    return df_vw


def cmf(df: pd.DataFrame, length: int = 21) -> pd.DataFrame:

    # Calculate Chaikin Money Flow

    df_cmf = df.copy()
    cmfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"])
    cmfmv = cmfm * df["volume"]
    df_cmf["cmf"] = cmfmv.rolling(window=length, min_periods=length).mean() / \
                    df["volume"].rolling(window=length, min_periods=length).mean()
    return df_cmf


def mfi(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:

    # Calculate Money Flow Index

    df_mfi = df.copy()
    change_up = ((df["close"] - df["close"].shift(1)) >= 0).astype(int)
    change_down = ((df["close"] - df["close"].shift(1)) <= 0).astype(int)
    upper = (df["volume"] * change_up * df["close"]).rolling(window=length, min_periods=length).sum()
    lower = (df["volume"] * change_down * df["close"]).rolling(window=length, min_periods=length).sum()
    df_mfi["mfi"] = 100.0 - (100.0 / (1.0 + upper / lower))

    return df_mfi


def ichimoku(df: pd.DataFrame, conv: int = 9, base: int = 26, span_b: int = 52):
    df['tenkan_sen'] = (df['high'].rolling(window=conv).max() + df['low'].rolling(window=conv).min()) / 2
    df['kijun_sen'] = (df['high'].rolling(window=base).max() + df['low'].rolling(window=base).min()) / 2
    df['senkou_span_a'] = (df['tenkan_sen'] + df['kijun_sen']) / 2
    df['senkou_span_b'] = (df['high'].rolling(window=span_b).max() + df['low'].rolling(window=span_b).min()) / 2
    df['chikou'] = df['close'].shift(base)
    return df


def analyze_ichimoku(df: pd.DataFrame, periods: Tuple[int, int, int]) -> Union[Tuple[IchimokuTrend, bool], None]:
        """Analyze Ichimoku signals for the current market state."""
        # Returns (0) trend and (1) bool value - trand is changing

        conv, base, span_b = periods
        df = ichimoku(df, conv=conv, base=base, span_b=span_b)

        if len(df) < 3:
            return None

        # Get the latest data points
        current = df.iloc[-1]
        prev = df.iloc[-2]

        current_trend = get_ichimoku_trend(current)
        prev_trend = get_ichimoku_trend(prev)

        global_current_trend = current_trend.value.replace("Strong", "").strip().lower()
        global_prev_trend = prev_trend.value.replace("Strong", "").strip().lower()

        return current_trend, global_prev_trend != global_current_trend


def get_ichimoku_trend(row: pd.Series) -> IchimokuTrend:
    # Price positioning
    price = row['close']

    # Tenkan-sen and Kijun-sen crossover detection
    tenkan_above_kijun = row['tenkan_sen'] > row['kijun_sen']
    tenkan_below_kijun = row['tenkan_sen'] < row['kijun_sen']

    # Price and Kumo (cloud) relationship
    cloud_top = max(row['senkou_span_a'], row['senkou_span_b'])
    cloud_bottom = min(row['senkou_span_a'], row['senkou_span_b'])

    price_above_kumo = price > cloud_top
    price_below_kumo = price < cloud_bottom
    price_in_kumo = not price_above_kumo and not price_below_kumo

    # Kumo characteristics
    kumo_bullish = row['senkou_span_a'] > row['senkou_span_b']
    kumo_bearish = row['senkou_span_a'] < row['senkou_span_b']

    # Price and Tenkan/Kijun relationship
    price_above_tenkan = price > row['tenkan_sen']
    price_below_tenkan = price < row['tenkan_sen']
    price_above_kijun = price > row['kijun_sen']
    price_below_kijun = price < row['kijun_sen']

    # Trend determination
    if price_above_kumo and kumo_bullish and tenkan_above_kijun:
        return IchimokuTrend.strong_bullish
    elif price_above_kumo or (price_above_tenkan and price_above_kijun):
        return IchimokuTrend.bullish
    elif price_below_kumo and kumo_bearish and tenkan_below_kijun:
        return IchimokuTrend.strong_bearish
    elif price_below_kumo or (price_below_tenkan and price_below_kijun):
        return IchimokuTrend.bearish
    return IchimokuTrend.consolidation
