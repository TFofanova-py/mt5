from datetime import datetime, timezone, timedelta
import json
import logging
from pair.wrapped_pair import WrappedPair
import MetaTrader5 as mt5
from time import sleep
import numpy as np
import argparse
import multiprocessing as mp


def aggregate_signals(ar: list, threshold: int = 3) -> int:
    threshold = min(len(ar), threshold)
    n_buy_signals = (np.array(ar) == 1).astype(int).sum()
    n_sell_signals = (np.array(ar) == -1).astype(int).sum()
    return 1 if n_buy_signals >= threshold else (-1 if n_sell_signals >= threshold else 0)


def continous_trading_pair(p_config: dict):
    tz = timezone(timedelta(hours=1))
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        filename=f"./History/logs/trading_log_{p_config['login']}_"
                                 f"{datetime.now(tz).date()}_{p_config['symbol']}.txt",
                        filemode="a")

    mt5.initialize(login=p_config['login'], password=p_config['password'],
                   server=p_config['server'], path=p_config['path'])
    p = WrappedPair(mt5, p_config)

    take_profit_benchmark = None

    while True:
        p.get_signals()

        signals = [aggregate_signals([a[1]["signal"] for a in v]) for k, v in p.last_state.items() if
                   k != "n_positions"]

        type_action = None
        if all([s == 1 for s in signals]) and len(p.positions) == 0:
            type_action = mt5.ORDER_TYPE_BUY
        elif all([s == -1 for s in signals]) and len(p.positions) == 0:
            type_action = mt5.ORDER_TYPE_SELL

        if type_action is not None:
            price = p.get_curr_price()
            response = p.create_position(price=price, type_action=type_action)

            info_bellinger = p.last_state[p.min_timeframe][1]["info"]
            take_profit_benchmark = info_bellinger["bollinger_bands_upper"] - info_bellinger["bollinger_bands_lower"]

            print(response)

        elif len(p.positions) > 0:
            take_profit, take_profit_benchmark = p.take_profit_signal(take_profit_benchmark)
            if take_profit or signals[-1] == 1:
                price = p.get_curr_price()
                responses = p.close_opened_position(price=price)
                print(responses)

        time_to_next_step = p.calc_time_to_next_step(signals)
        print(f"{p.symbol}: Sleep for {time_to_next_step // 60} minutes")
        sleep(time_to_next_step)


def error_callback(er):
    print(er)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    args = parser.parse_args()

    config = json.load(open(args.config_file))

    # tz = timezone(timedelta(hours=1))
    # logging.basicConfig(level=logging.INFO,
    #                     format='%(message)s',
    #                     filename=f"./History/logs/trading_log_{config['login']}_{datetime.now(tz).date()}.txt",
    #                     filemode="a")

    n_symbols = len(config["symbols"])
    pool = mp.Pool(n_symbols)

    for symbol in config["symbols"]:
        pair_config = {k: v for k, v in config.items() if k != "symbols"}
        pair_config.update({"symbol": symbol})

        pool.apply_async(continous_trading_pair, args=(pair_config,), error_callback=error_callback)

    pool.close()
    pool.join()
