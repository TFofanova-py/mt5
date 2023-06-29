import pandas as pd
from .ta_utils import donchian
from .yahoo_utils import get_yahoo_data
import argparse
from typing import Tuple


def get_ichimoku_signal(data: pd.DataFrame, ts: int = 9, tm: int = 26, tl: int = 52, offset: int = 26,
                        verbose: bool = False) -> Tuple[int, dict]:
    def strength(pos: int, uptrend: bool) -> int:
        if uptrend:
            return 1 if pos == 1 else (2 if pos == 0 else 3)
        return -1 if pos == -1 else (-2 if pos == 0 else -3)

    def cloudtrend(l1: float, l2: float) -> int:
        return 1 if l1 > l2 else -1

    ichimoku_df = data.copy()

    ichimoku_df["conversion"] = donchian(data, ts)
    ichimoku_df["base"] = donchian(data, tm)
    ichimoku_df["lead1"] = ichimoku_df[["conversion", "base"]].mean(axis=1, skipna=False)
    ichimoku_df["lead2"] = donchian(data, tl)
    ichimoku_df["cloud_top2"] = ichimoku_df[["lead1", "lead2"]].max(axis=1, skipna=False)
    ichimoku_df["cloud_bot2"] = ichimoku_df[["lead1", "lead2"]].min(axis=1, skipna=False)

    ichimoku_df["lead1_current"] = ichimoku_df["lead1"].shift(offset)
    ichimoku_df["lead2_current"] = ichimoku_df["lead2"].shift(offset)
    ichimoku_df["cloud_top"] = ichimoku_df[["lead1_current", "lead2_current"]].max(axis=1, skipna=False)
    ichimoku_df["cloud_bot"] = ichimoku_df[["lead1_current", "lead2_current"]].min(axis=1, skipna=False)
    ichimoku_df["base_position"] = ichimoku_df.apply(
        lambda x: 1 if x["base"] > x["cloud_top"] else (-1 if x["base"] < x["cloud_bot"] else 0),
        axis=1)
    ichimoku_df["base_breakout"] = ichimoku_df.apply(lambda x: strength(x["base_position"], x["close"] > x["base"]),
                                                     axis=1)
    ichimoku_df["cloud2_trend"] = ichimoku_df[["conversion", "base"]].apply(
        lambda x: cloudtrend(x["conversion"], x["base"]),
        axis=1)
    ichimoku_df["cloud2_top"] = ichimoku_df[["conversion", "base"]].max(axis=1, skipna=False)
    ichimoku_df["cloud2_bot"] = ichimoku_df[["conversion", "base"]].min(axis=1, skipna=False)
    ichimoku_df["cloud2_position"] = (
        ichimoku_df.apply(
            lambda x: 1 if x["cloud2_bot"] > x["cloud_top"] else (-1 if x["cloud2_top"] < x["cloud_bot"] else 0),
            axis=1)
    )
    ichimoku_df["cloud2_cross"] = ichimoku_df.apply(lambda x: strength(x["cloud2_position"], x["cloud2_trend"] == 1),
                                                    axis=1)
    ichimoku_df["lagging_lead1"] = ichimoku_df["lead1_current"].shift(offset)
    ichimoku_df["lagging_lead2"] = ichimoku_df["lead2_current"].shift(offset)
    ichimoku_df["lagging_cloud_top"] = ichimoku_df[["lagging_lead1", "lagging_lead2"]].max(axis=1, skipna=False)
    ichimoku_df["lagging_cloud_bot"] = ichimoku_df[["lagging_lead1", "lagging_lead2"]].min(axis=1, skipna=False)

    ichimoku_df["lagging_high"] = ichimoku_df["high"].shift(offset)
    ichimoku_df["lagging_low"] = ichimoku_df["low"].shift(offset)
    ichimoku_df["lagging_trend"] = (
        ichimoku_df.apply(
            lambda x: 1 if x["close"] > x["lagging_high"] else (-1 if x["close"] < x["lagging_low"] else 0),
            axis=1)
    )
    ichimoku_df["lagging_position"] = (
        ichimoku_df.apply(
            lambda x: 1 if x["close"] > x["cloud_top"] else (-1 if x["close"] < x["cloud_bot"] else 0), axis=1)
    )
    ichimoku_df["lagging_cross"] = ichimoku_df.apply(lambda x: strength(x["lagging_position"], x["lagging_trend"] == 1),
                                                     axis=1)
    ichimoku_df["cloud_breakout"] = ichimoku_df["lagging_position"]
    ichimoku_df["cloud_trend"] = ichimoku_df.apply(lambda x: cloudtrend(x["lead1"], x["lead2"]), axis=1)
    ichimoku_df["lead_cross"] = ichimoku_df.apply(lambda x: strength(x["cloud_breakout"], x["cloud_trend"] == 1),
                                                  axis=1)
    # signal
    indicators = ["base_breakout", "cloud2_cross", "lagging_cross", "cloud_breakout", "lead_cross"]
    signal_max = ichimoku_df.iloc[-1][indicators].astype(int).max()
    signal_min = ichimoku_df.iloc[-1][indicators].astype(int).min()
    signal = signal_max if signal_min > 0 else (signal_min if signal_max < 0 else 0)

    map_signals = {1: "Strong Up", 2: "Neutral Up", 3: "Weak Up",
                   -1: "Strong Down", -2: "Neutral Down", -3: "Weak Down", 0: "Consolidation"}

    info = {"time": ichimoku_df.index[-1],
            "Tenkan-Sen Price Cross": map_signals[ichimoku_df.iloc[-1]["cloud2_cross"]],
            "Kijun-Sen Price Cross": map_signals[ichimoku_df.iloc[-1]["base_breakout"]],
            "Chikou Span Price Cross": map_signals[ichimoku_df.iloc[-1]["lagging_cross"]],
            "Kumo Breakout": map_signals[ichimoku_df.iloc[-1]["cloud_breakout"]],
            "Kumo Twist": map_signals[ichimoku_df.iloc[-1]["lead_cross"]],
            "Status": map_signals[signal]}
    if verbose:
        print(info)

    return signal, info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yahoo_symbol", "-ys", type=str)
    parser.add_argument("--timeframe", "-tf", choices=["1m", "1h", "1d"])

    args = parser.parse_args()

    df = get_yahoo_data(symbol=args.yahoo_symbol, interval=args.timeframe, max_rows=500)
    ichimoku_signal = get_ichimoku_signal(df, verbose=True)
