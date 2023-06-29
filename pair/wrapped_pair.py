from datetime import datetime
import pytz
import logging
from .ichimoku import get_ichimoku_signal
import pandas as pd
import numpy as np
from .constants import NUM_ATTEMPTS_FETCH_PRICES, \
    HIGHEST_FIB, CENTRAL_HIGH_FIB, CENTRAL_LOW_FIB, LOWEST_FIB, MT5_TIMEFRAME
from typing import Tuple, List, Union
from .ta_utils import crossunder, crossover, rsi, bollinger_bands, pivotlevel, sharp_n_true
from .check_data import fix_missing


class WrappedPair:
    def __init__(self, broker, kwargs):
        self.broker = broker  # MT5
        self.symbol: str = kwargs["symbol"]  # symbol of the instrument MT5
        self.yahoo_symbol: str = kwargs.get("yahoo_symbol")  # symbol of the instrument on finance.yahoo.com

        self.deal_size: float = kwargs[
            "deal_size"] if "deal_size" in kwargs else 1.0  # deal size for opening position

        # stop level coefficient for opening position
        self.stop_coefficient: float = kwargs["stop_coefficient"] if "stop_coefficient" in kwargs else 0.98

        # allowed decrease of bollinger window, default 1%
        self.margin_take_profit: float = kwargs["margin_take_profit"] if "margin_take_profit" in kwargs else 0.99

        mt5_symbol_info = self.broker.symbol_info(self.symbol)
        self.trade_tick_size = mt5_symbol_info.trade_tick_size if mt5_symbol_info is not None else 0.1

        self.positions: list = []
        self.last_sell = None  # use only for dft signal

        minute_multiplier = {"w": 60 * 24 * 7, "d": 60 * 24, "h": 60, "m": 1}
        self.wraps = [x.update({"resolution": int(x["timeframe"][:-1]) * minute_multiplier[x["timeframe"][-1]]})
                      for x in kwargs["wraps"]]
        self.wraps = sorted(kwargs["wraps"], key=lambda x: x["resolution"], reverse=True)
        self.min_timeframe = self.wraps[-1]["timeframe"]

        self.last_state = {v["timeframe"]: [[a[0], {"signal": None, "info": {}}] for a in v["algorithms"]]
                           for v in self.wraps}
        self.last_state["n_positions"] = 0

    def get_historical_data(self, interval="1m", numpoints=500):
        i = 0

        try:
            mt5_timeframe = MT5_TIMEFRAME[interval]
        except KeyError as e:
            print(e)
            return None
        else:
            while i < NUM_ATTEMPTS_FETCH_PRICES:

                rates = self.broker.copy_rates_from_pos(self.symbol, mt5_timeframe, 0, numpoints)

                if rates is None:
                    i += 1
                    continue

                curr_prices = pd.DataFrame(rates)
                curr_prices["time"] = pd.to_datetime(curr_prices["time"], unit="s").dt.tz_localize(
                    pytz.timezone("Etc/GMT-3"))
                curr_prices.set_index("time", inplace=True)
                curr_prices = curr_prices[["close", "open", "high", "low"]]
                curr_prices = fix_missing(curr_prices, mandatory_cols=["close", "open", "high", "low"])
                return curr_prices

    @staticmethod
    def add_levels(df: pd.DataFrame, period: int = 24, highest_fib: float = HIGHEST_FIB,
                   central_high_fib: float = CENTRAL_HIGH_FIB,
                   central_low_fib: float = CENTRAL_LOW_FIB,
                   lowest_fib: float = LOWEST_FIB) -> pd.DataFrame:

        df = df.copy()

        df["hb"] = df["high"].rolling(window=period, min_periods=period).max()  # high border
        df["lb"] = df["low"].rolling(window=period, min_periods=period).min()  # low border
        df["dist"] = df["hb"] - df["lb"]  # range of the channel
        df["med"] = (df["lb"] + df["hb"]) / 2  # median of the channel

        df["hf"] = df["hb"] - highest_fib * df["dist"]  # highest fib
        df["chf"] = df["hb"] - central_high_fib * df["dist"]  # central high fib
        df["clf"] = df["hb"] - central_low_fib * df["dist"]  # central low fib
        df["lf"] = df["hb"] - lowest_fib * df["dist"]  # lowest fib

        return df

    def calc_time_to_next_step(self, signals: list) -> int:
        min_result = self.wraps[-1]["resolution"] * 60

        if len(self.positions) == 0:
            try:
                idx_wrap = signals.index(0)
                tf_wrap = self.wraps[idx_wrap]["timeframe"]

                result = self.wraps[idx_wrap]["resolution"] * 60 - (datetime.now(tz=pytz.timezone("Etc/GMT-3")) -
                                                                    self.last_state[tf_wrap][-1][1]["info"][
                                                                        "time"]).seconds
                return max(60, result)
            except ValueError:
                pass

        return min_result

    def get_curr_price(self) -> float:
        prices = self.get_historical_data(interval="1m", numpoints=2)
        return float(prices.iloc[-1]["close"])

    @staticmethod
    def get_rsi_signal(prices: pd.DataFrame,
                       period: int = 14, upper_threshold: int = 70, lower_threshold: int = 30) -> Tuple[int, dict]:
        rsi_ser = rsi(prices["close"], period=period)
        signal = 1 if rsi_ser[-1] < lower_threshold else (-1 if rsi_ser[-1] > upper_threshold else 0)
        info = {"time": rsi_ser.index[-1], "rsi": rsi_ser[-1]}
        return signal, info

    @staticmethod
    def get_bollinger_bands_signal(prices: pd.DataFrame, period: int = 200, std: float = 2.) -> Tuple[int, dict]:
        bb_df = bollinger_bands(prices["close"], period=period, std=std)
        signal = 1 if crossover(prices["close"], bb_df["lower"]) \
            else (-1 if crossunder(prices["close"], bb_df["upper"]) else 0)
        info = {"time": bb_df.index[-1],
                "bollinger_bands_upper": bb_df.iloc[-1]["upper"],
                "bollinger_bands_lower": bb_df.iloc[-1]["lower"]}
        return signal, info

    @staticmethod
    def get_support_resistance_signal(prices: pd.DataFrame,
                                      right_bars: int = 10, left_bars: int = 10, patience: int = 2) -> Tuple[int, dict]:

        window_size = left_bars + right_bars + 1
        support = prices["close"].rolling(window=window_size,
                                          min_periods=window_size).apply(pivotlevel,
                                                                         args=(right_bars,
                                                                               "low")).fillna(method="ffill")
        resistance = prices["close"].rolling(window=window_size,
                                             min_periods=window_size).apply(pivotlevel,
                                                                            args=(right_bars,
                                                                                  "high")).fillna(method="ffill")
        signal = 1 if sharp_n_true(pd.Series(prices["close"] > resistance), patience) else \
            (-1 if sharp_n_true(pd.Series(prices["close"] < support), patience) else 0)

        info = {"time": prices.index[-1], "support": support[-1], "resistance": resistance[-1]}
        return signal, info

    def get_signals(self, verbose: bool = True, save_history=True) -> None:

        for wrap in self.wraps:
            prices = self.get_historical_data(wrap["timeframe"])
            msg = None

            for i, (alg, params) in enumerate(wrap["algorithms"]):

                signal = 0
                info = {}
                self.positions = self.broker.positions_get(symbol=self.symbol)

                if alg == "ichimoku":
                    signal, info = get_ichimoku_signal(prices)

                    msg = f"{self.symbol}, {info['time']}, {wrap['timeframe']}-ichimoku signal is {signal}, status is {info['Status']}"

                elif alg == "rsi":
                    signal, info = self.get_rsi_signal(prices, period=params["period"],
                                                       lower_threshold=params["lower_threshold"],
                                                       upper_threshold=params["upper_threshold"])

                    msg = f"{self.symbol}, {info['time']}, {wrap['timeframe']}-RSI signal is {signal}, RSI is {info['rsi']}"

                elif alg == "bollinger_bands":
                    signal, info = self.get_bollinger_bands_signal(prices, period=params["period"], std=params["std"])

                    msg = f"{self.symbol}, {info['time']}, {wrap['timeframe']}-bollinger bands signal is {signal}, " \
                          f"bollinger_bands_upper is {info['bollinger_bands_upper']}, " \
                          f"bollinger_bands_lower is {info['bollinger_bands_lower']}"

                elif alg == "support_resistance":
                    signal, info = self.get_support_resistance_signal(prices, left_bars=params["left_bars"],
                                                                      right_bars=params["right_bars"])

                    msg = f"{self.symbol}, {info['time']}, {wrap['timeframe']}-support/resistance signal is {signal}, " \
                          f"support is {info['support']}, resistance is {info['resistance']}"

                else:
                    print(f"Algorithm {alg} is not implemented yet")

                if msg is not None:
                    logging.info(msg)

                    if verbose:
                        print(msg)

                self.last_state[wrap["timeframe"]][i][1] = {"signal": signal, "info": info}

        return None

    def take_profit_signal(self, prev_benchmark: float = None) -> Tuple[bool, Union[float, None]]:
        info_bellinger = self.last_state[self.min_timeframe][1]["info"]
        benchmark = info_bellinger["bollinger_bands_upper"] - info_bellinger["bollinger_bands_lower"]
        if benchmark < prev_benchmark * self.margin_take_profit:
            return True, None
        else:
            return False, benchmark

    def create_position(self, price: float, type_action: str, sl: float = None):
        if sl is None:
            sl = round(price * self.stop_coefficient, abs(int(np.log10(self.trade_tick_size))))

        request = {
            "action": self.broker.TRADE_ACTION_DEAL,  # for non market order TRADE_ACTION_PENDING
            "symbol": self.symbol,
            "volume": self.deal_size,
            "type": type_action,
            "price": price,
            "sl": sl,
            "comment": "python script open",
            "type_time": self.broker.ORDER_TIME_GTC,
            "type_filling": self.broker.ORDER_FILLING_IOC
        }

        # check order before placement
        check_result = self.broker.order_check(request)

        # if the order is incorrect
        if check_result.retcode != 0:
            # error codes are here: https://www.mql5.com/en/docs/constants/errorswarnings/enum_trade_return_codes
            print(check_result.retcode, check_result.comment)
            return None

        return self.broker.order_send(request)

    def close_opened_position(self, price: float, identifiers: List[int] = None) -> list:
        if identifiers is None:
            identifiers = [pos.identifier for pos in self.broker.positions_get(symbol=self.symbol)]

        responses = []
        for position in identifiers:
            request = {
                "action": self.broker.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": self.deal_size,
                "type": self.broker.ORDER_TYPE_SELL,
                "position": position,
                "price": price,
                "comment": "python script close",
                "type_time": self.broker.ORDER_TIME_GTC,
                "type_filling": self.broker.ORDER_FILLING_IOC
            }
            # check order before placement
            # check_result = broker.order_check(request)
            responses.append(self.broker.order_send(request))

        return responses

