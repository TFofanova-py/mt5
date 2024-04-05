import json
import os
from typing import List

import MetaTrader5 as mt5
import numpy as np
from datetime import datetime, timedelta, timezone
from time import sleep
import logging
from pair.multiindpair import MultiIndPair
from utils import wait_for_next_hour
import argparse


def create_position(broker, pair: MultiIndPair, sl: float = None):
    curr_price = pair.get_curr_price()

    if sl is None:
        sl = round(curr_price * pair.stop_coefficient, abs(int(np.log10(pair.trade_tick_size))))

    request = {
        "action": broker.TRADE_ACTION_DEAL,  # for non market order TRADE_ACTION_PENDING
        "symbol": pair.symbol,
        "volume": pair.deal_size,
        "type": broker.ORDER_TYPE_BUY,
        "price": curr_price,
        "sl": sl,
        "comment": "python script open",
        "type_time": broker.ORDER_TIME_GTC,
        "type_filling": broker.ORDER_FILLING_IOC
    }

    # check order before placement
    check_result = broker.order_check(request)

    # if the order is incorrect
    if check_result.retcode != 0:
        # error codes are here: https://www.mql5.com/en/docs/constants/errorswarnings/enum_trade_return_codes
        print(check_result.retcode, check_result.comment)
        return None

    return broker.order_send(request)


def close_opened_position(broker, pair, identifiers: List[int] = None) -> list:
    if identifiers is None:
        identifiers = [pos.identifier for pos in broker.positions_get(symbol=pair.symbol)]

    curr_price = pair.get_curr_price()

    responses = []
    for position in identifiers:
        request = {
            "action": broker.TRADE_ACTION_DEAL,
            "symbol": pair.symbol,
            "volume": pair.deal_size,
            "type": broker.ORDER_TYPE_SELL,
            "position": position,
            "price": curr_price,
            "comment": "python script close",
            "type_time": broker.ORDER_TIME_GTC,
            "type_filling": broker.ORDER_FILLING_IOC
        }
        # check order before placement
        # check_result = broker.order_check(request)
        responses.append(broker.order_send(request))

    return responses


def make_ichimoku_trade(broker, pair: MultiIndPair, verbose: bool = True):
    pass


def make_dft_trade(broker, pair, prev_trade_result, verbose=True):
    numpoints = prev_trade_result["numpoints"]
    prev_position = prev_trade_result["position"]

    pair.fetch_prices(broker, numpoints=numpoints)

    t_index = pair.prices.index[-1]

    # DFT algorithm
    pair.positions = broker.positions_get(symbol=pair.symbol)
    curr_dft_position = len(pair.positions)

    dft_signal = None
    if len(pair.positions) < prev_position:  # there was a stoploss
        dft_signal = -2

    info = {}
    dp_msg = None

    if dft_signal is None:
        dft_signal, info = pair.get_dft_signal(verbose=verbose)

    if dft_signal == -1 and pair.idx_down_periods > 0:
        pair.idx_down_periods = 0
        dp_msg = f"reset DP, DP = {pair.n_down_periods[pair.idx_down_periods]}"

    if dft_signal != 0:

        if dft_signal == 1 and (pair.is_multibuying_available or curr_dft_position == 0):
            resp_open = create_position(broker, pair, sl=info.get("upper_hf"))
            curr_dft_position += 1

            if resp_open is not None:
                logging.info(f"{pair.symbol}, {t_index}, buy, order {resp_open.order}")

        elif dft_signal in [-1, -3, -4] and curr_dft_position > 0:
            resp_close = close_opened_position(broker, pair, identifiers=info.get("identifiers"))
            pair.last_sell = t_index
            done_orders = [r.order for r in resp_close if r.retcode == 10009]
            logging.info(f"{pair.symbol}, {t_index}, sell, "
                         f"orders {done_orders}")
            curr_dft_position -= len(done_orders)

        if dft_signal in [-2, -4]:
            if pair.idx_down_periods < len(pair.n_down_periods) - 1:  # n_down_periods [3, 4, inf]
                pair.idx_down_periods = pair.idx_down_periods + 1
                dp_msg = f"increase DP, DP = {pair.n_down_periods[pair.idx_down_periods]}"

    # it doesn't work because of a delay on the broker side
    # curr_dft_position = len(broker.positions_get(symbol=pair.symbol))

    logging.info(f"{pair.symbol}, {t_index}, signal {dft_signal}, "
                 f"curr_dft_position {curr_dft_position}")
    if dp_msg is not None:
        logging.info(dp_msg)

    if verbose and dft_signal != 0:
        print(f"{t_index}, DFT signal: {dft_signal}, "
              f"price: {pair.prices.loc[t_index, 'close']}")
        if dp_msg is not None:
            print(dp_msg)

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

            pair_to_trade = MultiIndPair(config)
            if pair_to_trade is not None:

                if not args.immediately:
                    wait_for_next_hour(verbose=True)

                trade_const_instrument(mt5, pair_to_trade)
        mt5.shutdown()

