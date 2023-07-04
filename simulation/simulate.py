from typing import Tuple, Union
from datetime import datetime
import argparse
import logging
import time
import json
from pair.pair import Pair
from pair.constants import N_DOWN_PERIODS
from pair.yahoo_utils import get_yahoo_data


def was_stoploss(p: Pair, bal: dict) -> bool:
    # check for stoploss on the last step

    if bal["position"]["size"] > 0:
        stop_price = bal["position"]["stop_level"]
        if p.prices["close"].iloc[-1] < stop_price:
            return True

    return False


def simulation_step(p: Pair, bal: dict) -> Tuple[datetime, int, bool]:
    t_last = p.prices.index[-1]
    upper_timeframe_criterion = None

    signal = None

    if was_stoploss(p, bal):
        signal = -2

    if signal is None:
        signal, upper_timeframe_criterion = p.get_dft_signal(dft_period=p.dft_period, save_history=False, verbose=False)

    return t_last, signal, upper_timeframe_criterion


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


def run_simulate(**params) -> Union[Tuple[float, dict], None]:
    pair = Pair(params)

    actions_dict = {-2: "stop loss", -1: "sell", 0: "hold", 1: "buy"}

    df_history = get_yahoo_data(pair.yahoo_symbol)
    df_u_history = get_yahoo_data(pair.symbol, interval="1h") if pair.upper_timeframe_parameters is not None else None
    # df_history = pd.read_csv("../Tests/SP500_yahoo_data.csv", index_col=0)
    # df_history.index = pd.to_datetime(df_history.index)

    if df_history is not None:
        balance = {"cash": 0, "position": {"level": None, "size": 0, "stop_level": None}}

        last_u_dt = None
        upper_timeframe_criterion = None

        for curr_row in range(pair.dft_period * pair.resolution + 1, df_history.shape[0], pair.resolution):

            pair.prices = df_history.iloc[:curr_row]
            if df_u_history is not None:
                pair.u_prices = df_u_history[df_u_history.index < df_history.index[curr_row]]

            # if upper_timeframe_criterion is False and didn't change, skip the step
            if upper_timeframe_criterion is not None and \
                    not upper_timeframe_criterion and \
                    last_u_dt is not None and \
                    last_u_dt == pair.u_prices.index[-1]:
                continue

            t_index, dft_signal, upper_timeframe_criterion = simulation_step(pair, balance)
            last_u_dt = pair.u_prices.index[-1] if pair.u_prices is not None else None

            pair = update_pair(pair, balance, dft_signal)
            balance = update_balance(pair, balance, dft_signal)

            if not (dft_signal == 0 and balance['position']['size'] == 0):
                logging.getLogger(__name__).info(f"{pair.symbol}, {t_index}, "
                                                 f"action {actions_dict[dft_signal]}, balance {balance}")
                print(f"{params}, {t_index}, "
                      f"action {actions_dict[dft_signal]}, balance {balance}")

        if balance["position"]["size"] > 0:
            balance["cash"] += balance["position"]["level"]

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
