import logging
import pandas as pd
import numpy as np
from .constants import MT5_TIMEFRAME
from typing import Literal, List
from .ta_utils import (
    rsi, macd, momentum, cci, obv,
    stk, vwmacd, cmf, mfi,
    pivotlow, pivothigh)


class MultiIndPair:

    def __init__(self, broker, kwargs):

        try:
            self.broker = broker
            self.symbol: str = kwargs["symbol"]  # symbol of the instrument MT5
            self.yahoo_symbol: str = kwargs.get("yahoo_symbol")  # symbol of the instrument on finance.yahoo.com

            # time steps for MT5 in minutes, they are different when open / close position
            self.resolution_set: dict = kwargs.get("resolution", {"open": 3, "close": 3})
            self.resolution: int = self.resolution_set["open"]

            self.deal_size: float = float(kwargs.get("deal_size", 1.0))  # volume for opening position must be a float
            self.devaition: int = kwargs.get("deviation",
                                             50)  # max number of points to squeeze when open / close position

            # stop level coefficient for opening position
            self.stop_coefficient: float = kwargs.get("stop_coefficient", 0.9995)

            mt5_symbol_info = self.broker.symbol_info(self.symbol)
            self.trade_tick_size = mt5_symbol_info.trade_tick_size if mt5_symbol_info is not None else 0.1

            self.positions: list = []

            # pivot periods, they are different when open / close position
            self.pivot_period_set: dict = kwargs.get("pivot_period", {"open": 5, "close": 5})
            self.pivot_period: int = self.pivot_period_set["open"]

            self.searchdiv = "Regular"  # "Regular/Hidden, Hidden
            self.min_number_of_divergence: dict = kwargs.get("min_number_of_divergence",
                                                             {"entry": 1,
                                                              "exit_sl": 1,
                                                              "exit_tp": 1})
            self.max_pivot_points: int = kwargs.get("max_pivot_points", 10)
            self.max_bars_to_check: int = kwargs.get("max_bars_to_check", 100)
            self.dont_wait_for_confirmation: bool = kwargs.get("dont_wait_for_confirmation", True)
            self.indicators: dict = kwargs["entry"]
            self.direction: Literal["low-long", "high-short", "bi"] = kwargs.get("direction", "low-long")
            self.exit_strategy: str = kwargs.get("exit", "default")

            self.last_divergence = {"top": None, "bottom": None}

        except TypeError as e:
            logging.error(f"Error: {e}")
            print(f"Error: {e}")

    def set_parameters_by_position(self):
        if self.positions is None or len(self.positions) == 0:
            self.resolution = self.resolution_set["open"]
            self.pivot_period = self.pivot_period_set["open"]
        else:
            self.resolution = self.resolution_set["close"]
            self.pivot_period = self.pivot_period_set["close"]

    def get_historical_data(self, numpoints: int = None):

        if numpoints is None:
            numpoints = self.max_bars_to_check

        try:
            if 60 <= self.resolution < 60 * 24 and self.resolution % 60 == 0:
                interval = MT5_TIMEFRAME[str(self.resolution // 60) + "h"]
            elif self.resolution == 60 * 24:
                interval = MT5_TIMEFRAME["1d"]
            elif self.resolution == 60 * 24 * 7:
                interval = MT5_TIMEFRAME["1wk"]
            else:
                interval = MT5_TIMEFRAME[str(self.resolution) + "m"]

        except KeyError:
            msg = f"{self.resolution} minutes is not a standard MetaTrade5 timeframe, choose another resolution"
            print(msg)
        else:
            rates = self.broker.copy_rates_from_pos(self.symbol, interval, 0, numpoints)

            if rates is None:
                print(f"{self.symbol}: Can't get the historical data")
                return None

            curr_prices = pd.DataFrame(rates)
            curr_prices["time"] = pd.to_datetime(curr_prices["time"], unit="s", utc=True)
            curr_prices.set_index("time", inplace=True)
            curr_prices = curr_prices[["close", "open", "high", "low", "tick_volume"]]
            curr_prices = curr_prices.rename({"tick_volume": "volume"}, axis=1)

            return curr_prices

    @staticmethod
    def arrived_divergence(src, close, startpoint, length, np_func):
        virtual_line_src = np.linspace(src.iloc[-length - 1], src.iloc[-startpoint - 1],
                                       length - startpoint)
        virtual_line_close = np.linspace(close.iloc[-length - 1], close.iloc[-startpoint - 1],
                                         length - startpoint)

        return all(np_func(src.iloc[-length - 1: -startpoint - 1], virtual_line_src)) and \
               all(np_func(close.iloc[-length - 1: -startpoint - 1], virtual_line_close))

    def divergence_length(self, src: pd.Series, close: pd.Series,
                          pivot_vals: np.array, pivot_positions: np.array,
                          mode: Literal["positive_regular", "negative_regular",
                                        "positive_hidden", "negative_hidden"]):

        def is_suspected():
            func_src = np.greater if mode in ["positive_regular", "negative_hidden"] else np.less
            func_close = np.less if mode in ["positive_regular", "negative_hidden"] else np.greater

            return func_src(src.iloc[-startpoint - 1], src.iloc[-length - 1]) and \
                   func_close(close.iloc[-startpoint - 1], pivot_vals.iloc[-x - 1])

        divlen = 0

        confirm_func = np.greater if mode in ["positive_regular", "positive_hidden"] else np.less
        scr_or_close_confirm = confirm_func(src.iloc[-1], src.iloc[-2]) and confirm_func(close.iloc[-1], close.iloc[-2])

        if self.dont_wait_for_confirmation or scr_or_close_confirm:
            startpoint = 0 if self.dont_wait_for_confirmation else 1

            for x in range(0, min(len(pivot_positions), self.max_pivot_points)):
                length = src.index[-1] - pivot_positions.iloc[-x - 1] + self.pivot_period

                # if we reach non valued array element or arrived 101. or previous bars then we don't search more
                if pivot_positions.iloc[-x - 1] == 0 or length > self.max_bars_to_check - 1:
                    break

                if length > 5 and is_suspected():
                    arrived_func = np.greater_equal if mode in ["positive_regular", "positive_hidden"] \
                        else np.less_equal

                    arrived = self.arrived_divergence(src, close, startpoint, length, arrived_func)

                    if arrived:
                        divlen = length
                        break

        return divlen

    def calculate_divs(self, indicator: pd.Series, close: pd.Series,
                       pl_vals: np.array, pl_positions: np.array,
                       ph_vals: np.array, ph_positions: np.array) -> np.array:
        divs = np.zeros(4, dtype=int)

        if self.searchdiv in ["Regular", "Regular/Hidden"]:
            divs[0] = self.divergence_length(indicator, close, pl_vals, pl_positions, "positive_regular")
            divs[1] = self.divergence_length(indicator, close, ph_vals, ph_positions, "negative_regular")

        if self.searchdiv in ["Hidden", "Regular/Hidden"]:
            divs[2] = self.divergence_length(indicator, close, pl_vals, pl_positions, "positive_hidden")
            divs[3] = self.divergence_length(indicator, close, ph_vals, ph_positions, "negative_hidden")

        return divs

    def count_divergence(self, data: pd.DataFrame) -> dict:

        ind_ser: List[pd.Series] = []
        indices: List[str] = []

        for ind_name, ind_params in self.indicators.items():

            if ind_name == "rsi":
                ind_ser.append(rsi(data, period=ind_params["rsi_length"])["rsi"].reset_index(drop=True))
                indices.append("rsi")

            elif ind_name == "macd":
                macd_df = macd(data, **ind_params)
                ind_ser.append(macd_df["macd"].reset_index(drop=True))
                ind_ser.append(macd_df["hist"].reset_index(drop=True))
                indices.extend(["macd", "deltamacd"])

            elif ind_name == "momentum":
                ind_ser.append(momentum(data, **ind_params)["momentum"].reset_index(drop=True))
                indices.append("momentum")

            elif ind_name == "cci":
                ind_ser.append(cci(data, **ind_params)["cci"].reset_index(drop=True))
                indices.append("cci")

            elif ind_name == "obv":
                ind_ser.append(obv(data)["obv"].reset_index(drop=True))
                indices.append("obv")

            elif ind_name == "stk":
                ind_ser.append(stk(data, **ind_params)["stk"].reset_index(drop=True))
                indices.append("stk")

            elif ind_name == "vwmacd":
                ind_ser.append(vwmacd(data, **ind_params)["vwmacd"].reset_index(drop=True))
                indices.append("vwmacd")

            elif ind_name == "cmf":
                ind_ser.append(cmf(data, **ind_params)["cmf"].reset_index(drop=True))
                indices.append("cmf")

            elif ind_name == "mfi":
                ind_ser.append(mfi(data, **ind_params)["mfi"].reset_index(drop=True))
                indices.append("mfi")

        all_divergences = pd.DataFrame(np.zeros((len(ind_ser), 4)), index=indices)

        pl_vals, pl_positions = pivotlow(data["close"], self.pivot_period)
        ph_vals, ph_positions = pivothigh(data["close"], self.pivot_period)

        for i, curr_ind_ser in enumerate(ind_ser):
            all_divergences.iloc[i, :] = self.calculate_divs(curr_ind_ser[-self.max_bars_to_check:],
                                                             data["close"][-self.max_bars_to_check:],
                                                             pl_vals, pl_positions,
                                                             ph_vals, ph_positions)

        n_indicators = all_divergences.shape[0]

        # div_types: 0 - bottom (div0 and div2), 1 - top (div1 and div3)
        div_types = (np.arange(4).reshape(1, -1) % 2) * np.ones((n_indicators, 1)).astype(int)

        top_mask = (div_types == 1)
        bottom_mask = (div_types == 0)

        div_signals = pd.DataFrame(np.zeros((n_indicators, 2)),
                                   index=indices, columns=["top", "bottom"])
        div_signals.iloc[:, 0] = np.any((all_divergences > 0) * top_mask, axis=1).astype(int)
        div_signals.iloc[:, 1] = np.any((all_divergences > 0) * bottom_mask, axis=1).astype(int)

        # update last_divergence and return
        for t in ["top", "bottom"]:
            if div_signals[t].sum() > 0:
                # if it was a pivot high/ low after last top/ bottom divergence
                new_divergence = True
                if self.last_divergence[t] is not None and self.last_divergence[t] >= data.index[0]:

                    idx = data.index.get_loc(self.last_divergence[t])

                    new_divergence = (pl_positions.iloc[-1] > idx) if t == "bottom" else (ph_positions.iloc[-1] > idx)

                if new_divergence:
                    self.last_divergence[t] = data.index[-1]
                    return {"top": div_signals["top"].sum(),
                            "bottom": div_signals["bottom"].sum()}

        return {"top": 0, "bottom": 0}

    @staticmethod
    def was_price_goes_up(data: pd.DataFrame) -> bool:
        assert "close" in data.columns
        assert data.shape[0] > 1
        if data["close"].iloc[-1] > data["close"].iloc[-2]:
            return True
        return False

    def create_position(self, price: float, type_action: str, sl: float = None):
        if sl is None:
            if type_action == 0:
                sl = round(price * self.stop_coefficient, abs(int(np.log10(self.trade_tick_size))))
            else:
                sl = round(price * (2. - self.stop_coefficient), abs(int(np.log10(self.trade_tick_size))))

        request = {
            "action": self.broker.TRADE_ACTION_DEAL,  # for non market order TRADE_ACTION_PENDING
            "symbol": self.symbol,
            "volume": self.deal_size,
            "deviation": self.devaition,
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
            print(check_result.retcode, check_result.comment, request)
            return None

        return self.broker.order_send(request)

    def close_opened_position(self, price: float, type_action: str, identifiers: List[int] = None) -> list:
        if identifiers is None:
            identifiers = [pos.identifier for pos in self.broker.positions_get(symbol=self.symbol)]

        responses = []
        for position in identifiers:
            request = {
                "action": self.broker.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": self.deal_size,
                "deviation": self.devaition,
                "type": type_action,
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

    def modify_sl(self, new_sls: List[float], identifiers: List[int] = None) -> list:
        if identifiers is None:
            identifiers = [pos.identifier for pos in self.broker.positions_get(symbol=self.symbol)]

        assert new_sls is not None
        assert len(identifiers) == len(new_sls)

        responses = []
        for position, new_sl in zip(identifiers, new_sls):
            request = {
                "action": self.broker.TRADE_ACTION_SLTP,
                "symbol": self.symbol,
                "sl": new_sl,
                "position": position
            }
            # check order before placement
            # check_result = broker.order_check(request)
            responses.append(self.broker.order_send(request))

        return responses
