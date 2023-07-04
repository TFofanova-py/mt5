import os
import MetaTrader5 as mt5
import numpy as np
from datetime import datetime, timedelta, timezone
from time import sleep
import logging
from pair.pair import Pair
from pair.constants import N_DOWN_PERIODS
import argparse
import json


def create_position(broker, pair):
    curr_price = pair.get_curr_price()
    request = {
        "action": broker.TRADE_ACTION_DEAL,
        "symbol": pair.symbol,
        "volume": pair.deal_size,
        "type": broker.ORDER_TYPE_BUY,
        "price": curr_price,
        "sl": round(curr_price * pair.stop_coefficient, abs(int(np.log10(pair.trade_tick_size)))),
        "comment": "python script open",
        "type_time": broker.ORDER_TIME_GTC,
        "type_filling": broker.ORDER_FILLING_IOC
    }

    # check order before placement
    check_result = broker.order_check(request)

    # if the order is incorrect
    if check_result.retcode != 0:
        # error codes here: https://www.mql5.com/en/docs/constants/errorswarnings/enum_trade_return_codes
        print(check_result.retcode, check_result.comment)
        return None

    return broker.order_send(request)


def close_opened_position(broker, pair):
    position = broker.positions_get(symbol=pair.symbol)[-1]
    curr_price = pair.get_curr_price()

    request = {
        "action": broker.TRADE_ACTION_DEAL,
        "symbol": pair.symbol,
        "volume": pair.deal_size,
        "type": broker.ORDER_TYPE_SELL,
        "position": position.identifier,
        "price": curr_price,
        "comment": "python script close",
        "type_time": broker.ORDER_TIME_GTC,
        "type_filling": broker.ORDER_FILLING_IOC
    }
    return broker.order_send(request)


def make_dft_trade(broker, pair, prev_trade_result):
    numpoints = prev_trade_result["numpoints"]
    prev_position = prev_trade_result["position"]

    pair.fetch_prices(broker, numpoints=numpoints)

    t_index = pair.prices.index[-1]

    # DFT algorithm
    try:
        curr_dft_position = len(broker.positions_get(symbol=pair.symbol))
    except TypeError:
        curr_dft_position = 0

    if curr_dft_position < prev_position:  # was stoploss
        dft_signal = -2
        if pair.idx_down_periods < len(N_DOWN_PERIODS) - 1:  # n_down_periods [3, 4, inf]
            pair.idx_down_periods = pair.idx_down_periods + 1
        logging.info(f"idx_n_down_periods {pair.idx_down_periods}")
    else:
        dft_signal, _ = pair.get_dft_signal(dft_period=pair.dft_period, verbose=True)

    logging.info(f"{pair.symbol}, {t_index}, signal {dft_signal}, curr_dft_position {curr_dft_position}")

    if dft_signal != 0:

        if dft_signal == 1 and (pair.is_multibuying_available or curr_dft_position == 0):
            # and pair.idx_down_periods != len(N_DOWN_PERIODS) - 1: - this condition is in get_dft_signal
            resp_open = create_position(broker, pair)

            if resp_open is not None:
                curr_dft_position += 1
                logging.info(f"{pair.symbol}, {t_index}, buy, order {resp_open.order}")

        elif dft_signal == -1:
            pair.idx_down_periods = 0
            logging.info(f"idx_down_periods {pair.idx_down_periods}")

            if curr_dft_position > 0:
                resp_close = close_opened_position(broker, pair)
                pair.last_sell = t_index
                curr_dft_position -= 1
                logging.info(f"{pair.symbol}, {t_index}, sell, order {resp_close.order}")

    return {"numpoints": 2 * pair.resolution + 1, "position": curr_dft_position}


def print_opened_pos(broker, pair):
    opened_pos = broker.positions_get(symbol=pair.symbol)
    if opened_pos is not None:
        logging.info(f"You have {len(opened_pos)} opened positions in {pair.symbol}")


def trade_const_instrument(broker, pair):
    print_opened_pos(broker, pair)
    trade_result = {"numpoints": pair.resolution * (pair.dft_period + 1), "position": 0}

    while True:
        # script works 24*7
        try:
            trade_result = make_dft_trade(broker, pair, trade_result)
        except Exception as e:
            print(e)

        sleep(pair.resolution * 60)


def wait_for_next_hour(verbose=False) -> None:
    now = datetime.now()
    seconds_to_next_hour = 60 * 60 - now.minute * 60 - now.second + 1

    msg = f"waiting for {seconds_to_next_hour} seconds for the next hour"
    logging.info(msg)
    if verbose:
        print(msg)

    sleep(seconds_to_next_hour)


def check_strategy_config(conf: dict) -> bool:
    # check if all required parameters are in config

    for req_param in ["symbol"]:
        if req_param not in conf:
            msg = f"Parameter {req_param} is required in config.json"
            logging.error(msg)
            print(msg)
            return False
    return True


if __name__ == '__main__':
    tz = timezone(timedelta(hours=1))

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("--immediately", "-i", action="store_true")
    args = parser.parse_args()

    config = json.load(open(args.config_file))

    if not os.path.exists("./History/logs/"):
        os.makedirs("./History/logs/")

    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        filename=f"./History/logs/trading_log_{config['login']}_{datetime.now(tz).date()}.txt",
                        filemode="a")

    try:
        mt5.initialize(login=config['login'], password=config['password'],
                       server=config['server'], path=config['path'])

    except Exception as e:
        logging.info(f"Create session error: {e}")
    else:
        if check_strategy_config(config):
            logging.info(f"Using instrument {config.get('symbol')}")

            pair_to_trade = Pair(config)
            if pair_to_trade is not None:

                if not args.immediately:
                    wait_for_next_hour(verbose=True)

                trade_const_instrument(mt5, pair_to_trade)
        mt5.shutdown()

