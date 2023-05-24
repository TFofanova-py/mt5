from typing import Tuple, Union
from datetime import datetime
import argparse
import logging
import numpy as np

from pair.pair import Pair, parse_du
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


def run_simulate(symbol="ES=F", resolution=3, deal_size=1.0, stop_coefficient=0.995,
                 limit=None, dft_period=34, highest_fib=0.220, lowest_fib=0.800,
                 down_periods="[3, 4]", is_multibuying_available=False,
                 upper_timeframe_parameters=None) -> Union[Tuple[float, dict], None]:
    pair = Pair(symbol=symbol,
                resolution=resolution,
                deal_size=deal_size,
                stop_coef=stop_coefficient,
                limit_coef=limit,
                dft_period=dft_period,
                highest_fib=highest_fib,
                lowest_fib=lowest_fib,
                n_down_periods=list(map(int, down_periods[1:-1].split(','))) + [np.inf],
                upper_timeframe_parameters=parse_du(upper_timeframe_parameters, sep=","),
                is_multibuying_available=is_multibuying_available
                )

    params = {"symbol": symbol, "resolution": resolution, "deal_size": deal_size,
              "stop_coefficient": stop_coefficient, "limit": limit, "dft_period": dft_period,
              "highest_fib": highest_fib, "lowest_fib": lowest_fib, "down_periods": down_periods,
              "is_multibuying_available": is_multibuying_available,
              "upper_timeframe_parameters": upper_timeframe_parameters}

    actions_dict = {-2: "stop loss", -1: "sell", 0: "hold", 1: "buy"}

    df_history = get_yahoo_data(pair.symbol)
    df_u_history = get_yahoo_data(pair.symbol, interval="1h") if upper_timeframe_parameters is not None else None
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
    parser.add_argument("--multibuying_avail", "-m", default=False, action="store_true",
                        help="buying is available even if there is an opened position")
    parser.add_argument("--upper_timeframe_parameters", "-du", type=str, default=None,
                        help="upper timeframe (hours) for buying in blue zone")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        filename=f"../History/logs/simulate_log_{datetime.now().date()}_yh{args.yahoo_symbol}"
                                 f"_r{args.resolution}_s{args.deal_size}_st{args.stop_coefficient}"
                                 f"_l{args.limit}_d{args.dft_period}_hf{args.highest_fib}"
                                 f"_lf{args.lowest_fib}"
                                 f"_dp{args.down_periods[1: -1].replace(',', '_').replace(' ', '')}"
                                 f"_m{args.multibuying_avail}"
                                 f"_du{args.upper_timeframe_parameters}.txt",
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
                               is_multibuying_available=args.multibuying_avail,
                               upper_timeframe_parameters=list(args.upper_timeframe_parameters[1:-1].split(", "))
                               )

    print(f"Total yield: {total_yield}")
