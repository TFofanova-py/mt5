import json
import time
from typing import Tuple, Union
from datetime import datetime
import logging
from itertools import islice
import numpy as np

from pair.pair import Pair
from pair.constants import N_DOWN_PERIODS
from pair.yahoo_utils import get_yahoo_data
from collections import namedtuple
import argparse

Position = namedtuple("Position", ["identifier", "time", "price", "size"])


def was_stoploss(p: Pair, bal: dict) -> bool:
    # check for stoploss on the last step

    if bal["position"]["size"] > 0:
        stop_price = bal["position"]["stop_level"]
        if p.prices["close"].iloc[-1] < stop_price:
            return True

    return False


def simulation_step(p: Pair, bal: dict) -> Tuple[datetime, int, dict]:
    t_last = p.prices.index[-1]
    info = {}

    signal = None

    if was_stoploss(p, bal):
        signal = -2

    if signal is None:
        signal, info = p.get_dft_signal(save_history=False)

    return t_last, signal, info


def update_pair(p: Pair, bal: dict, signal: int) -> Pair:
    if signal in [-2, -4]:
        if p.idx_down_periods < len(N_DOWN_PERIODS) - 1:  # n_down_periods [3, 4, inf]
            p.idx_down_periods = p.idx_down_periods + 1

    if signal == -1 and p.idx_down_periods > 0:
        p.idx_down_periods = 0
        print(f"DP = {p.n_down_periods[p.idx_down_periods]}")

    if signal < 0 and bal['position']['size'] > 0:
        p.last_sell = p.prices.index[-1]
        p.positions = []
        print(f"DP = {p.n_down_periods[p.idx_down_periods]}")

    if signal == 1 and (p.positions is None or len(p.positions) == 0):
        t_index = p.prices.index[-1]
        trade_time = int(datetime.timestamp(t_index))
        p.positions = [Position(identifier=trade_time, time=trade_time,
                                size=p.deal_size, price=p.get_curr_price())]

    return p


def update_balance(p: Pair, bal: dict, signal: int, info: dict = None) -> dict:
    t_last = p.prices.index[-1]
    curr_position = bal["position"]["size"]

    if signal == 1 and curr_position == 0:
        bal["cash"] -= p.get_curr_price() * p.deal_size
        bal["position"]["level"] = p.prices.loc[t_last, "close"]
        bal["position"]["size"] = p.deal_size
        assert not (p.stop_coefficient == "hf" and info is None)
        bal["position"]["stop_level"] = p.stop_coefficient * bal["position"]["level"] \
            if type(p.stop_coefficient) == float else info.get("upper_hf")

    elif signal < 0 and curr_position > 0:
        if bal["position"]["level"] > p.get_curr_price():
            bal["n_neg_inrow"] += 1
        else:
            bal["n_neg_inrow"] = 0

        bal["cash"] += p.get_curr_price() * p.deal_size
        bal["position"] = {"level": None, "size": 0, "stop_level": None, "duration": 0}

    elif signal >= 0 and curr_position > 0:
        # bal["position"]["level"] = p.get_curr_price()
        bal["position"]["duration"] += 1

    return bal


def early_stop(bal: dict, p: Pair, max_neg_inrow: int = 5, min_total_yield: float = -0.5) -> bool:
    if bal["n_neg_inrow"] == max_neg_inrow:
        return True

    if bal["position"]["size"] > 0:
        return False

    return bal["cash"] / (p.get_curr_price() * p.deal_size) * 100 < min_total_yield


def run_simulate(**params) -> Union[Tuple[float, dict], None]:
    pair = Pair(params)

    actions_dict = {-4: "down trend",
                    -3: "unclear trend",
                    -2: "stop loss",
                    -1: "sell",
                    0: "hold",
                    1: "buy"}

    df_history = get_yahoo_data(pair.yahoo_symbol)
    df_u_history = get_yahoo_data(pair.yahoo_symbol,
                                  interval="1h") if pair.upper_timeframe_parameters is not None else None
    # df_history = pd.read_csv("../Tests/SP500_yahoo_data.csv", index_col=0)
    # df_history.index = pd.to_datetime(df_history.index)

    if df_history is not None:
        balance = {"cash": 0, "position": {"level": None, "size": 0, "stop_level": None, "duration": 0}, "n_neg_inrow": 0}

        it_periods = iter(range(pair.dft_period * pair.resolution + 1, df_history.shape[0], pair.resolution))
        for curr_row in it_periods:

            pair.prices = df_history.iloc[:curr_row]
            if df_u_history is not None:
                pair.u_prices = df_u_history[df_u_history.index < df_history.index[curr_row]]

            # t_index, dft_signal, info = simulation_step(pair, balance)
            t_index = pair.prices.index[-1]
            info = pair.get_blue_dft_signal()
            upper_timeframe_criterion = info.get("upper_timeframe_criterion")

            # if upper_timeframe_criterion is False and aren't opened positions, skip the hour
            if not upper_timeframe_criterion and balance["position"]["size"] == 0:

                # skip several periods in a row till next hour
                delta_minute = 60 - (t_index - pair.u_prices.index[-1]).seconds // 60
                n_skip_periods = max(0, delta_minute // pair.resolution - 1)
                if n_skip_periods > 0:
                    try:
                        next(islice(it_periods, n_skip_periods - 1, n_skip_periods))
                    except StopIteration:
                        break
            else:
                t_index, dft_signal, info = simulation_step(pair, balance)

                pair = update_pair(pair, balance, dft_signal)
                balance = update_balance(pair, balance, dft_signal, info)

                if not (dft_signal == 0 and balance['position']['size'] == 0):
                    logging.getLogger(__name__).info(f"{pair.symbol}, {t_index}, "
                                                     f"action {actions_dict[dft_signal]}, balance {balance}")
                    print(f"{t_index}, "
                          f"action {actions_dict[dft_signal]}, balance {balance}")

            if early_stop(balance, pair,
                          max_neg_inrow=params["max_neg_inrow"], min_total_yield=params["min_total_yield"]):
                print("Early stop ", params)
                break

        if balance["position"]["size"] > 0:
            balance["cash"] += balance["position"]["level"] * balance["position"]["size"]

        tot_yield = round(balance["cash"] / df_history.loc[pair.last_sell]["close"] * 100, 1) \
            if pair.last_sell is not None else 0

        return tot_yield, params

    else:
        print("Something wrong with historical data")
        return None


if __name__ == '__main__':
    start = time.time()

    strategy_config = json.load(open("../config.json"))

    pair = Pair(strategy_config)

    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        filename=f"../History/logs/simulate_log_{datetime.now().date()}_yh{pair.yahoo_symbol}"
                                 f"_r{pair.resolution}_s{pair.deal_size}"
                                 f"_st{pair.stop_coefficient}"
                                 f"_l{pair.limit_coefficient}_d{pair.dft_period}_hf{pair.highest_fib}"
                                 f"_lf{pair.lowest_fib}"
                                 f"_dp{pair.n_down_periods}"
                                 f"_m{pair.is_multibuying_available}"
                                 f"_du{pair.upper_timeframe_parameters}.txt",
                        filemode="w")

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--max_neg_inrow", type=int, default=3,
                        help="Max negative trades in a row to break simulation")
    parser.add_argument("-y", "--min_total_yield", type=float, default=-0.5,
                        help="Min percentage of total yeild to break simulation")
    args = parser.parse_args()

    ext_config = {"max_neg_inrow": args.max_neg_inrow, "min_total_yield": args.min_total_yield}
    ext_config.update(strategy_config)
    total_yield, _ = run_simulate(**ext_config)

    print(f"Total yield: {total_yield}, time: {time.time() - start}")
