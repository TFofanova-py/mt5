import datetime
import logging
import numpy as np
import pytz
import os.path

import MetaTrader5 as mt5
import pandas as pd
from .constants import NUM_ATTEMPTS_FETCH_PRICES, \
    HIGHEST_FIB, CENTRAL_HIGH_FIB, CENTRAL_LOW_FIB, LOWEST_FIB, \
    N_DOWN_PERIODS, MIN_PRICE_HIST_PERIOD, MT5_TIMEFRAME
from .check_data import fix_missing
from .yahoo_utils import get_yahoo_data
from typing import Literal, Tuple, Union
from .ta_utils import crossunder


class Pair:
    def __init__(self, kwargs):

        try:
            self.symbol: str = kwargs["symbol"]  # symbol of the instrument MT5
            self.yahoo_symbol: str = kwargs.get("yahoo_symbol")  # symbol of the instrument on finance.yahoo.com
            self.resolution: int = kwargs["resolution"] if "resolution" in kwargs else 3  # time step for MT5 in minutes

            # source of historical data
            self.data_source: Literal["mt5", "yahoo"] = kwargs["data_source"] if "data_source" in kwargs else "yahoo"
            if self.data_source == "yahoo":
                assert self.yahoo_symbol is not None, "yahoo_symbol mustn't be None if data_source is 'yahoo'"
            self.prices = None
            self.data_file: str = f"./History/{self.symbol}_rawdata.csv"

            self.deal_size: float = kwargs[
                "deal_size"] if "deal_size" in kwargs else 1.0  # deal size for opening position

            # stop level coefficient for opening position
            self.stop_coefficient: Union[float, Literal["hf"]] = kwargs[
                "stop_coefficient"] if "stop_coefficient" in kwargs else 0.9995

            self.limit_coefficient: float = kwargs.get("limit_coefficient")  # take profit for opening position

            mt5_symbol_info = mt5.symbol_info(self.symbol)
            self.trade_tick_size = mt5_symbol_info.trade_tick_size if mt5_symbol_info is not None else 0.1

            self.dft_period: int = kwargs[
                "dft_period"] if "dft_period" in kwargs else 24  # period for making dtf signal
            self.idx_down_periods: int = 0
            self.last_sell: datetime = None
            self.highest_fib: float = kwargs["highest_fib"] if "highest_fib" in kwargs else HIGHEST_FIB  # hf level
            self.central_high_fib: float = kwargs[
                "central_high_fib"] if "central_high_fib" in kwargs else CENTRAL_HIGH_FIB
            self.lowest_fib: float = kwargs["lowest_fib"] if "lowest_fib" in kwargs else LOWEST_FIB  # lf level
            self.central_low_fib: float = kwargs["central_low_fib"] if "central_low_fib" in kwargs else CENTRAL_LOW_FIB

            # down periods for buy signal
            self.n_down_periods: tuple = tuple(
                kwargs["down_periods"] + [np.inf]) if "down_periods" in kwargs else N_DOWN_PERIODS

            # buying is available even if there is an opened position
            self.is_multibuying_available: bool = kwargs[
                "multibuying_available"] if "multibuying_available" in kwargs else False

            # upper timeframe (hours) for buying in blue zone
            self.upper_timeframe_parameters: Tuple[int, int, float] = kwargs.get("upper_timeframe_parameters")
            assert self.upper_timeframe_parameters is None or 0 < self.upper_timeframe_parameters[2] < 1, \
                "The third upper_timeframe_parameter must be in (0, 1)"
            self.u_prices = None
            self.long_up_trend = None

            self.unclear_trend_periods: int = kwargs[
                "unclear_trend_periods"] if "unclear_trend_periods" in kwargs else 30
            self.down_trend_periods: int = kwargs["down_trend_periods"] if "down_trend_periods" in kwargs else 5
            self.positions: list = []

        except TypeError as e:
            logging.error(f"Error: {e}")
            print(f"Error: {e}")

    def save_raw_data(self, data):
        new_data = data
        if os.path.exists(self.data_file):
            with open(self.data_file) as f:
                last_record_dt_str = f.readlines()[-1].split(",")[0][:-6]
                last_record_dt = datetime.datetime.strptime(last_record_dt_str, "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=pytz.UTC)
                new_data = data[data.index > last_record_dt]

        new_data.to_csv(self.data_file, mode="a", header=False)

    def get_historical_data(self, session, interval="1m",  numpoints=MIN_PRICE_HIST_PERIOD):
        i = 0
        rates = None

        while i < NUM_ATTEMPTS_FETCH_PRICES:

            if self.data_source == "mt5":
                rates = session.copy_rates_from_pos(self.symbol, MT5_TIMEFRAME[interval], 0, numpoints)
            elif self.data_source == "yahoo":
                rates = get_yahoo_data(self.yahoo_symbol, interval=interval)

            if rates is None:
                i += 1
                continue

            if self.data_source == "yahoo":
                return rates

            curr_prices = pd.DataFrame(rates)
            curr_prices["time"] = pd.to_datetime(curr_prices["time"], unit="s", utc=True)
            curr_prices.set_index("time", inplace=True)
            curr_prices = curr_prices[["close", "open", "high", "low"]]
            return curr_prices

    @staticmethod
    def update_prices(data: pd.DataFrame, new_values: pd.DataFrame):
        if data is None:
            data = fix_missing(new_values, ["close", "open", "high", "low"])
            return data

        last_dt = data.index[-1]
        curr_prices = new_values[new_values.index > last_dt]
        if len(curr_prices) > 0:
            data = pd.concat([data, curr_prices], axis=0)
        return fix_missing(data, ["close", "open", "high", "low"])

    def fetch_prices(self, session, numpoints=MIN_PRICE_HIST_PERIOD):

        # fetch current prices
        curr_prices = self.get_historical_data(session, numpoints=numpoints)
        self.save_raw_data(curr_prices)

        self.prices = self.update_prices(self.prices, curr_prices)

        # fetch upper timeframe prices if its needed
        if self.upper_timeframe_parameters is not None:
            assert len(self.upper_timeframe_parameters) == 3, f"upper_timeframe_parameters must have 3 parameters"

            u_resolution = int(self.upper_timeframe_parameters[0])
            u_dft_period = int(self.upper_timeframe_parameters[1])
            u_numpoints = u_resolution * u_dft_period + 1
            curr_u_prices = self.get_historical_data(session, interval="1h", numpoints=u_numpoints)
            self.u_prices = self.update_prices(self.u_prices, curr_u_prices)
        return None

    def get_curr_price(self):
        return self.prices.iloc[-1]["close"]

    def range_len(self, n_ranges=1):
        return n_ranges * self.trade_tick_size

    @staticmethod
    def crossover(ts_1, ts_2):  # ts_1 crossovers ts_2
        if (ts_1.iloc[-2] < ts_2.iloc[-2]) and (ts_1.iloc[-1] > ts_2.iloc[-1]):
            return True
        return False

    @staticmethod
    def crossunder(ts_1, ts_2):  # ts_1 crossunders ts_2
        if (ts_1.iloc[-2] > ts_2.iloc[-2]) and (ts_1.iloc[-1] < ts_2.iloc[-1]):
            return True
        return False

    def add_levels(self, df: pd.DataFrame, period: int = None, highest_fib: float = None) -> pd.DataFrame:

        df = df.copy()

        # for upper_dft_df period and highest_fib aren't the attributes of self
        period = self.dft_period if period is None else period
        highest_fib = self.highest_fib if highest_fib is None else highest_fib

        df["hb"] = df["high"].rolling(window=period, min_periods=period).max()  # high border
        df["lb"] = df["low"].rolling(window=period, min_periods=period).min()  # low border
        df["dist"] = df["hb"] - df["lb"]  # range of the channel
        df["med"] = (df["lb"] + df["hb"]) / 2  # median of the channel

        df["hf"] = df["hb"] - highest_fib * df["dist"]  # highest fib
        df["chf"] = df["hb"] - self.central_high_fib * df["dist"]  # central high fib
        df["clf"] = df["hb"] - self.central_low_fib * df["dist"]  # central low fib
        df["lf"] = df["hb"] - self.lowest_fib * df["dist"]  # lowest fib

        return df

    @staticmethod
    def apply_resolution(data: pd.DataFrame,  resolution: int = 1, interval: Literal["minute", "hour"] = "minute"):
        if resolution == 1:
            return data.copy()

        res_df = pd.DataFrame(columns=data.columns)

        # takes every resolution from the end of data
        res_df["close"] = data["close"].iloc[-1::-resolution]
        res_df["open"] = data["open"].rolling(window=resolution, min_periods=resolution).apply(
            lambda x: x[0])
        res_df["low"] = data["low"].rolling(window=resolution, min_periods=resolution).min()
        res_df["high"] = data["high"].rolling(window=resolution, min_periods=resolution).max()
        return res_df.sort_index()

    def get_blue_dft_signal(self, verbose=True) -> Tuple[int, dict]:
        criterion = True
        result = {}

        if self.upper_timeframe_parameters is not None:

            u_resolution, u_dft_period, u_hf = self.upper_timeframe_parameters

            upper_dft_df = self.apply_resolution(self.u_prices, u_resolution, interval="hour")
            upper_dft_df = self.add_levels(upper_dft_df, u_dft_period, highest_fib=u_hf)
            criterion = (upper_dft_df["close"] > upper_dft_df["hf"]).values[-1]

            if criterion and self.stop_coefficient == "hf":
                result["upper_hf"] = upper_dft_df.iloc[-1]["hf"]

            # logging and verbose
            if self.long_up_trend is None:
                msg = f"in blue zone" if criterion else f"out of blue zone"
                logging.info(f"{self.symbol}, {upper_dft_df.index[-1]}, {u_resolution}h-timeframe is {msg}")
                if verbose:
                    print(self.symbol, upper_dft_df.index[-1], f"{u_resolution}h-timeframe is {msg}")
                self.long_up_trend = criterion

            else:
                u_evupin = criterion and not self.long_up_trend
                u_evupout = not criterion and self.long_up_trend

                if u_evupin:  # upper timeframe enters up trend
                    logging.info(f"{self.symbol}, {upper_dft_df.index[-1]}, {u_resolution}h-timeframe enters blue zone")
                    if verbose:
                        print(self.symbol, upper_dft_df.index[-1], f"{u_resolution}h-timeframe enters blue zone")
                    self.long_up_trend = True

                if u_evupout:  # upper timeframe leaves up trend
                    logging.info(f"{self.symbol}, {upper_dft_df.index[-1]}, {u_resolution}h-timeframe leaves blue zone")
                    if verbose:
                        print(self.symbol, upper_dft_df.index[-1], f"{u_resolution}h-timeframe leaves blue zone")
                    self.long_up_trend = False

        result["upper_timeframe_criterion"] = criterion

        return int(criterion), result

    def get_dft_signal(self, verbose=False, save_history=True) -> Tuple[int, dict]:
        def sharp_n_true(ser: pd.Series, n: int):
            if ser.shape[0] < n:
                return False
            elif ser.shape[0] == n and all(ser):
                return True
            return all(ser.iloc[-n:]) and not ser.iloc[-n - 1]

        if self.prices is not None:

            # data must be without missing
            assert self.prices.isna().sum().sum() == 0, "There is missing in pair.prices"

            dft_df = self.apply_resolution(self.prices, self.resolution)

            # channel
            dft_df = self.add_levels(dft_df)

            # entry markers
            dft_df["close_1"] = dft_df["close"].shift(1)
            dft_df["True_range"] = dft_df.apply(lambda x: max(x["high"] - x["low"],
                                                              abs(x["high"] - x["close_1"]),
                                                              abs(x["low"] - x["close_1"])), axis=1)
            dft_df["Atr"] = dft_df["True_range"].rolling(window=self.dft_period,
                                                         min_periods=self.dft_period).mean()
            dft_df["Tol"] = dft_df["Atr"] * 0.2  # tolerance for placing triangles and prediction candles at borders

            signal = 0
            # evupin = crossover(dft_df["close"], dft_df["hf"])  # market enters up trend
            evupout = crossunder(dft_df["close"], dft_df["hf"])  # market leaves up trend

            # if use information from upper timeframe (it has to be in blue zone - up trend)
            blue_dft_signal, info = self.get_blue_dft_signal(verbose=verbose)

            if evupout:
                signal = -1
            dft_df["lftrue"] = dft_df["close"] < dft_df["lf"]

            n_down_periods = self.n_down_periods[self.idx_down_periods]
            if self.idx_down_periods != len(self.n_down_periods) - 1 and \
                    sharp_n_true(dft_df["lftrue"], n_down_periods) and \
                    (self.last_sell is None or self.last_sell < dft_df.index[-n_down_periods]) and \
                    blue_dft_signal:
                # последние n_down_period были оранжевыми и с последней продажи прошло более n_down_periods периодов,
                # обертка в синей зоне

                signal = 1

            # if after unclear_trend_periods price isn't in blue zone, close the position
            if len(self.positions) > 0:
                durations = self.get_position_durations()
                durations = durations[durations == self.unclear_trend_periods]
                identifier_to_sell = durations.index[0] if durations.shape[0] > 0 else None

                # if curr price isn't in blue zone and position is opened - all previous prices are out of the blue zone
                if identifier_to_sell is not None and dft_df.iloc[-1]["close"] < dft_df.iloc[-1]["hf"]:
                    signal = -3
                    info["identifiers"] = [int(identifier_to_sell)]

            for position in self.positions:
                t = datetime.datetime.fromtimestamp(position.time, tz=pytz.UTC)
                if sharp_n_true(dft_df[dft_df.index > t]["lftrue"], self.down_trend_periods):
                    signal = -4
                    if "identifiers" in info:
                        info["identifiers"].append(int(position.identifier))
                    else:
                        info["identifiers"] = [int(position.identifier)]

            if save_history:
                dft_df.to_csv(f"History/dft_{datetime.datetime.now().date()}_{self.resolution}min.csv")

            return signal, info

    def get_position_durations(self) -> pd.Series:
        assert len(self.positions) > 0, "There aren't any opened positions"
        opened_pos_times = pd.DataFrame(self.positions,
                                        columns=self.positions[0]._asdict().keys())[["identifier",
                                                                                "time"]].set_index("identifier")
        opened_pos_times = pd.to_datetime(opened_pos_times["time"], unit="s", utc=True)
        curr_time = self.prices.index[-1]
        return pd.Series(opened_pos_times).apply(lambda t: (curr_time - t).seconds / (60 * self.resolution)).astype(int)

    def cross_stop_loss(self, session, stop_price):
        self.fetch_prices(session)
        return self.prices["close"].iloc[-1] < stop_price
