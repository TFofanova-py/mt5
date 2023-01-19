import datetime
import pytz
import os.path

import MetaTrader5 as mt5
import pandas as pd
from constants import NUM_ATTEMPTS_FETCH_PRICES, \
    HIGHEST_FIB, CENTRAL_HIGH_FIB, CENTRAL_LOW_FIB, LOWEST_FIB, \
    N_DOWN_PERIODS, MIN_PRICE_HIST_PERIOD
from check_data import fix_missing
from yahoo_utils import get_yahoo_data


class Pair:
    def __init__(self, symbol=None, yahoo_symbol=None,
                 resolution=3, data_source="mt5", deal_size=1,
                 stop_coef=0.9995, limit_coef=None, dft_period=24,
                 highest_fib=HIGHEST_FIB, central_high_fib=CENTRAL_HIGH_FIB,
                 lowest_fib=LOWEST_FIB, central_low_fib=CENTRAL_LOW_FIB,
                 n_down_periods=N_DOWN_PERIODS):

        self.symbol = symbol
        self.yahoo_symbol = yahoo_symbol
        self.resolution = resolution
        self.data_source = data_source
        if self.data_source == "yahoo":
            assert self.yahoo_symbol is not None, "yahoo_symbol mustn't be None if data_source is 'yahoo'"
        self.prices = None
        self.data_file = f"./History/{self.symbol}_rawdata.csv"
        self.deal_size = deal_size
        self.stop_coefficient = stop_coef
        self.limit_coefficient = limit_coef

        mt5_symbol_info = mt5.symbol_info(symbol)
        self.trade_tick_size = mt5_symbol_info.trade_tick_size if mt5_symbol_info is not None else 0.1

        self.dft_period = dft_period
        self.idx_down_periods = 0
        self.last_sell = None
        self.highest_fib = highest_fib
        self.central_high_fib = central_high_fib
        self.lowest_fib = lowest_fib
        self.central_low_fib = central_low_fib
        self.n_down_periods = n_down_periods

    def save_raw_data(self, data):
        new_data = data
        if os.path.exists(self.data_file):
            with open(self.data_file) as f:
                last_record_dt_str = f.readlines()[-1].split(",")[0][:-6]
                last_record_dt = datetime.datetime.strptime(last_record_dt_str, "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=pytz.UTC)
                new_data = data[data.index > last_record_dt]

        new_data.to_csv(self.data_file, mode="a", header=False)

    def get_historical_data(self, session, numpoints=MIN_PRICE_HIST_PERIOD):
        i = 0
        rates = None

        while i < NUM_ATTEMPTS_FETCH_PRICES:

            if self.data_source == "mt5":
                rates = session.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M1, 0, numpoints)
            elif self.data_source == "yahoo":
                rates = get_yahoo_data(self.yahoo_symbol)

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

    def fetch_prices(self, session, numpoints=MIN_PRICE_HIST_PERIOD):
        curr_prices = self.get_historical_data(session, numpoints)
        self.save_raw_data(curr_prices)

        if self.prices is None:
            self.prices = fix_missing(curr_prices, ["close", "open", "high", "low"])
            return

        last_dt = self.prices.index[-1]
        curr_prices = curr_prices[curr_prices.index > last_dt]
        if len(curr_prices) > 0:
            self.prices = pd.concat([self.prices, curr_prices], axis=0)
        self.prices = fix_missing(self.prices, ["close", "open", "high", "low"])
        return

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

    def add_channels(self, df, period):
        df = df.copy()
        df["hb"] = df["high"].rolling(window=period, min_periods=period).max()  # high border
        df["lb"] = df["low"].rolling(window=period, min_periods=period).min()  # low border
        df["dist"] = df["hb"] - df["lb"]  # range of the channel
        df["med"] = (df["lb"] + df["hb"]) / 2  # median of the channel

        df["hf"] = df["hb"] - self.highest_fib * df["dist"]  # highest fib
        df["chf"] = df["hb"] - self.central_high_fib * df["dist"]  # central high fib
        df["clf"] = df["hb"] - self.central_low_fib * df["dist"]  # central low fib
        df["lf"] = df["hb"] - self.lowest_fib * df["dist"]  # lowest fib
        return df

    def apply_resolution(self):
        if self.resolution == 1:
            return self.prices.copy()

        res_df = pd.DataFrame(columns=self.prices.columns)
        while self.prices.index[0].minute % self.resolution != 1:
            self.prices = self.prices.drop(self.prices.index[0])
        res_df["close"] = self.prices["close"].iloc[self.resolution - 1::self.resolution]
        res_df["open"] = self.prices["open"].rolling(window=self.resolution, min_periods=self.resolution).apply(
            lambda x: x[0])
        res_df["low"] = self.prices["low"].rolling(window=self.resolution, min_periods=self.resolution).min()
        res_df["high"] = self.prices["high"].rolling(window=self.resolution, min_periods=self.resolution).max()
        return res_df

    def get_dft_signal(self, dft_period=24, verbose=False, save_history=True):
        if self.prices is not None:

            # data must be without missing
            assert self.prices.isna().sum().sum() == 0, "There is missing in pair.prices"

            dft_df = self.apply_resolution()

            # channel
            dft_df = self.add_channels(dft_df, dft_period)

            # entry markers
            dft_df["close_1"] = dft_df["close"].shift(1)
            dft_df["True_range"] = dft_df.apply(lambda x: max(x["high"] - x["low"],
                                                              abs(x["high"] - x["close_1"]),
                                                              abs(x["low"] - x["close_1"])), axis=1)
            dft_df["Atr"] = dft_df["True_range"].rolling(window=dft_period,
                                                         min_periods=dft_period).mean()
            dft_df["Tol"] = dft_df["Atr"] * 0.2  # tolerance for placing triangles and prediction candles at borders

            signal = 0
            # evupin = crossover(dft_df["close"], dft_df["hf"])  # market enters up trend
            evupout = self.crossunder(dft_df["close"], dft_df["hf"])  # market leaves up trend
            if evupout:
                signal = -1
            dft_df["lftrue"] = dft_df["close"] < dft_df["lf"]

            n_down_periods = self.n_down_periods[self.idx_down_periods]
            if self.idx_down_periods != len(self.n_down_periods) - 1 and \
                    all(dft_df["lftrue"].iloc[-n_down_periods:]) and \
                    not dft_df["lftrue"].iloc[-n_down_periods - 1] and \
                    (self.last_sell is None or self.last_sell < dft_df.index[-n_down_periods]):
                # последние n_down_period были оранжевыми и с последней продажи прошло более n_down_periods периодов

                signal = 1

            if verbose and signal != 0:
                t_ind = dft_df.index[-1]
                print(f"{t_ind}, DFT signal: {signal}, "
                      f"price: {dft_df.loc[t_ind, 'close']}, "
                      f"levels: (hb: {dft_df.loc[t_ind, 'hb']}, "
                      f"hf: {dft_df.loc[t_ind, 'hf']}, "
                      f"lf: {dft_df.loc[t_ind, 'lf']},"
                      f"lb: {dft_df.loc[t_ind, 'lb']})")

            if save_history:
                dft_df.to_csv(f"History/dft_{datetime.datetime.now().date()}_{self.resolution}min.csv")
            return signal

    def cross_stop_loss(self, session, stop_price):
        self.fetch_prices(session)
        return self.prices["close"].iloc[-1] < stop_price
