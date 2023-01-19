from typing import Tuple, Union
from datetime import datetime
import argparse
import logging
import numpy as np
import pandas as pd

from pair import Pair
from constants import N_DOWN_PERIODS
from yahoo_utils import get_yahoo_data


def was_stoploss(p: Pair, bal: dict) -> bool:
    # check for stoploss on the last step

    if bal["position"]["size"] > 0:
        stop_price = bal["position"]["stop_level"]
        if p.prices["close"].iloc[-1] < stop_price:
            return True

    return False


def simulation_step(p: Pair, bal: dict) -> Tuple[datetime, int]:
    t_last = p.prices.index[-1]

    signal = None

    if was_stoploss(p, bal):
        signal = -2

    if signal is None:
        signal = p.get_dft_signal(dft_period=p.dft_period, save_history=False)

    return t_last, signal


def update_pair(p: Pair, bal: dict, signal: int) -> Pair:
    if signal == -2:
        if p.idx_down_periods < len(N_DOWN_PERIODS) - 1:  # n_down_periods [3, 4, inf]
            p.idx_down_periods = p.idx_down_periods + 1

    if signal == -1:
        p.idx_down_periods = 0

    if signal < 0 and bal['position']['size'] > 0:
        p.last_sell = p.prices.index[-1]

    return p


def update_balance(p: Pair, bal: dict, signal: int) -> dict:
    t_last = p.prices.index[-1]
    curr_position = bal['position']['size']

    if signal == 1 and curr_position == 0:
        bal["cash"] -= p.prices.loc[t_last, "close"] * p.deal_size
        bal["position"]["level"] = p.prices.loc[t_last, "close"]
        bal["position"]["size"] = p.deal_size
        bal["position"]["stop_level"] = p.stop_coefficient * bal["position"]["level"]

    elif signal < 0 and curr_position > 0:
        bal["cash"] += p.prices.loc[t_last, "close"] * p.deal_size
        bal["position"] = {"level": None, "size": 0, "stop_level": None}

    elif signal == 0 and curr_position > 0:
        bal["position"]["level"] = p.prices.loc[t_last, "close"]

    return bal


def run_simulate(symbol="ES=F", resolution=3, deal_size=1.0, stop_coefficient=0.995,
                 limit=None, dft_period=34, highest_fib=0.220, lowest_fib=0.800,
                 down_periods="[3, 4]") -> Union[Tuple[float, dict], None]:
    pair = Pair(symbol=symbol,
                resolution=resolution,
                deal_size=deal_size,
                stop_coef=stop_coefficient,
                limit_coef=limit,
                dft_period=dft_period,
                highest_fib=highest_fib,
                lowest_fib=lowest_fib,
                n_down_periods=list(map(int, down_periods[1:-1].split(','))) + [np.inf]
                )

    actions_dict = {-2: "stop loss", -1: "sell", 0: "hold", 1: "buy"}

    df_history = get_yahoo_data(pair.symbol)
    # df_history = pd.read_csv("../Tests/SP500_yahoo_data.csv", index_col=0)
    # df_history.index = pd.to_datetime(df_history.index)

    if df_history is not None:
        balance = {"cash": 0, "position": {"level": None, "size": 0, "stop_level": None}}

        for curr_row in range(pair.dft_period * pair.resolution + 1, df_history.shape[0], pair.resolution):

            pair.prices = df_history.iloc[:curr_row]

            t_index, dft_signal = simulation_step(pair, balance)

            pair = update_pair(pair, balance, dft_signal)
            balance = update_balance(pair, balance, dft_signal)

            if not (dft_signal == 0 and balance['position']['size'] == 0):
                logging.info(f"{pair.symbol}, {t_index}, action {actions_dict[dft_signal]}, balance {balance}")

        if balance["position"]["size"] > 0:
            balance["cash"] += balance["position"]["level"]

        tot_yield = round(balance["cash"] / df_history.loc[pair.last_sell]["close"] * 100, 1)
        params = {"symbol": symbol, "resolution": resolution, "deal_size": deal_size,
                  "stop_coefficient": stop_coefficient, "limit": limit, "dft_period": dft_period,
                  "highest_fib": highest_fib, "lowest_fib": lowest_fib, "down_periods": down_periods}

        return tot_yield, params

    else:
        print("Something wrong with historical data")
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--yahoo_symbol", "-yh", default="ES=F", help="yahoo ticker to get historical data")
    parser.add_argument("--resolution", "-r", default=3, type=int, help="time step for IG Trading")
    parser.add_argument("--deal_size", "-s", default=1.0, type=float, help="deal size for opening position")
    parser.add_argument("--stop_coefficient", "-st", type=float, default=0.995,
                        help="stop level coefficient for opening position")
    parser.add_argument("--limit", "-l", default=None, type=float, help="take profit for opening position")
    parser.add_argument("--dft_period", "-d", type=int, default=34, help="period for make dtf signal")
    parser.add_argument("--highest_fib", "-hf", type=float, default=0.220, help="hf level")
    parser.add_argument("--lowest_fib", "-lf", type=float, default=0.800, help="lf level")
    parser.add_argument("--down_periods", "-dp", type=str, default="[3, 4]", help="down periods for buy signal")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        filename=f"./History/logs/simulate_log_{datetime.now().date()}_yh{args.yahoo_symbol}"
                                 f"_r{args.resolution}_s{args.deal_size}_st{args.stop_coefficient}"
                                 f"_l{args.limit}_d{args.dft_period}_hf{args.highest_fib}"
                                 f"_lf{args.lowest_fib}"
                                 f"_dp{args.down_periods[1: -1].replace(',', '_').replace(' ', '')}.txt",
                        filemode="w")

    total_yield = run_simulate(symbol=args.yahoo_symbol,
                               resolution=args.resolution,
                               deal_size=args.deal_size,
                               stop_coefficient=args.stop_coefficient,
                               limit=args.limit,
                               dft_period=args.dft_period,
                               highest_fib=args.highest_fib,
                               lowest_fib=args.lowest_fib,
                               down_periods=args.down_periods,
                               )

    print(f"Total yield: {total_yield}")
