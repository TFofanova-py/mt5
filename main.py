import os
import time

import MetaTrader5 as mt5
import numpy as np
from config_mt5 import Config
from datetime import datetime, timedelta, timezone
from time import sleep
import logging
from pair import Pair
from constants import MIN_PRICE_HIST_PERIOD, N_DOWN_PERIODS
import argparse


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
        "type_filling": broker.ORDER_FILLING_FOK
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
        "type_filling": broker.ORDER_FILLING_FOK
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
        dft_signal = pair.get_dft_signal(dft_period=pair.dft_period)

    logging.info(f"{pair.symbol}, {t_index}, signal {dft_signal}, curr_dft_position {curr_dft_position}")

    if dft_signal != 0:

        if dft_signal == 1 and curr_dft_position == 0 and pair.idx_down_periods != len(N_DOWN_PERIODS) - 1:
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


def trade_const_instrument(broker, pair, tzone):
    print_opened_pos(broker, pair)
    trade_result = {"numpoints": pair.resolution * (pair.dft_period + 1), "position": 0}

    while True:
        # script works 24*7
        try:
            trade_result = make_dft_trade(broker, pair, trade_result)
        except Exception as e:
            print(e)

        sleep(pair.resolution * 60)


if __name__ == '__main__':
    tz = timezone(timedelta(hours=1))

    if not os.path.exists("./History/logs/"):
        os.makedirs("./History/logs/")

    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", "-i", default="[SP500]", help="symbol of the instrument MT5")
    parser.add_argument("--yahoo_symbol", "-ys", default="^GSPC", help="symbol of the instrument on finance.yahoo.com")
    parser.add_argument("--resolution", "-r", default=3, type=int, help="time step for MT5 in minutes")
    parser.add_argument("--data_source", "-ds", default="yahoo", choices=["mt5", "yahoo"], help="source of historical data")
    parser.add_argument("--deal_size", "-s", default=1.0, type=float, help="deal size for opening position")
    parser.add_argument("--stop_coefficient", "-st", type=float, default=0.998,
                        help="stop level coefficient for opening position")
    parser.add_argument("--limit", "-l", default=None, type=float, help="take profit for opening position")
    parser.add_argument("--dft_period", "-d", type=int, default=39, help="period for make dtf signal")
    parser.add_argument("--highest_fib", "-hf", type=float, default=0.220, help="hf level")
    parser.add_argument("--lowest_fib", "-lf", type=float, default=0.800, help="lf level")
    parser.add_argument("--server", "-srv", type=str, default=Config.server, help="MT5 server")
    parser.add_argument("--login", "-lgn", type=int, default=Config.login, help="MT5 login")
    parser.add_argument("--password", "-psw", type=str, default=Config.password, help="MT5 password")
    parser.add_argument("--path", "-pth", type=str, default=Config.path, help="MT5 path")
    parser.add_argument("--down_periods", "-dp", type=str, default="[3, 4]", help="down periods for buy signal")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        filename=f"./History/logs/trading_log_{args.login}_{datetime.now(tz).date()}.txt",
                        filemode="a")

    try:
        mt5.initialize(login=args.login, password=args.password, server=args.server, path=args.path)

    except Exception as e:
        logging.info(f"Create session error: {e}")
    else:
        if args.instrument is not None:
            logging.info(f"Using instrument from {args.instrument}")
            pair_to_trade = Pair(symbol=args.instrument,
                                 yahoo_symbol=args.yahoo_symbol,
                                 resolution=args.resolution,
                                 data_source=args.data_source,
                                 deal_size=args.deal_size,
                                 stop_coef=args.stop_coefficient,
                                 limit_coef=args.limit,
                                 dft_period=args.dft_period,
                                 highest_fib=args.highest_fib,
                                 lowest_fib=args.lowest_fib,
                                 n_down_periods=list(map(int, args.down_periods[1:-1].split(','))) + [np.inf])
            if pair_to_trade is not None:
                trade_const_instrument(mt5, pair_to_trade, tz)
        else:
            logging.info(f"Choosing pair by bot is not available now")
        mt5.shutdown()

