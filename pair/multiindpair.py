import logging
import pandas as pd
import numpy as np
from .constants import MT5_TIMEFRAME
from typing import Literal, Union, List, Tuple
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
            self.resolution: int = kwargs.get("resolution", 3)  # time step for MT5 in minutes

            self.deal_size: float = float(kwargs.get("deal_size", 1.0))  # volume for opening position must be a float

            # stop level coefficient for opening position
            self.stop_coefficient: float = kwargs.get("stop_coefficient", 0.9995)

            mt5_symbol_info = self.broker.symbol_info(self.symbol)
            self.trade_tick_size = mt5_symbol_info.trade_tick_size if mt5_symbol_info is not None else 0.1

            self.positions: list = []

            self.pivot_period: int = kwargs.get("pivot_period", 5)
            self.searchdiv = "Regular"  # "Regular/Hidden, Hidden
            self.min_number_of_divergence: int = kwargs.get("min_number_of_divergence", 1)
            self.max_pivot_points: int = kwargs.get("max_pivot_points", 10)
            self.max_bars_to_check: int = kwargs.get("max_bars_to_check", 100)
            self.dont_wait_for_confirmation: bool = kwargs.get("dont_wait_for_confirmation", True)
            self.indicators: dict = kwargs["entry"]
            self.direction: Literal["low-long", "high-short", "bi"] = kwargs.get("direction", "low-long")
            self.exit_strategy: str = kwargs.get("exit", "default")

            self.last_bottom_divergence = None
            self.last_top_divergence = None

        except TypeError as e:
            logging.error(f"Error: {e}")
            print(f"Error: {e}")

    def get_historical_data(self, numpoints: int = None):

        if numpoints is None:
            numpoints = self.max_bars_to_check

        try:
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

    def positive_regular_positive_hidden_divergence(self, src: pd.Series, close: pd.Series,
                                                    pl_vals: np.array, pl_positions: np.array,
                                                    cond: int = 1):
        # function to check positive regular or negative hidden divergence
        # cond == 1 = > positive_regular, cond == 2 = > negative_hidden

        divlen = 0
        # if pl_positions.iloc[-1] == pl_positions.iloc[-2]:
        #     return divlen

        # if indicators higher than last value and close price is higher than las close
        if self.dont_wait_for_confirmation or src.iloc[-1] > src.iloc[-2] or close.iloc[-1] > close.iloc[-2]:
            startpoint = 0 if self.dont_wait_for_confirmation else 1

            for x in range(0, min(len(pl_positions), self.max_pivot_points)):
                length = src.index[-1] - pl_positions.iloc[-x - 1] + self.pivot_period

                # if we reach non valued array element or arrived 101. or previous bars then we don't search more
                if pl_positions.iloc[-x - 1] == 0 or length > self.max_bars_to_check - 1:
                    break

                if length > 5 and \
                        ((cond == 1 and src.iloc[-startpoint - 1] > src.iloc[-length - 1] and close.iloc[
                            -startpoint - 1] < pl_vals.iloc[-x - 1]) or \
                         (cond == 2 and src.iloc[-startpoint - 1] < src.iloc[-length - 1] and close.iloc[
                             -startpoint - 1] > pl_vals.iloc[-x - 1])):

                    virtual_line_src = np.linspace(src.iloc[-length - 1], src.iloc[-startpoint - 1],
                                                   length - startpoint)
                    virtual_line_close = np.linspace(close.iloc[-length - 1], close.iloc[-startpoint - 1],
                                                     length - startpoint)
                    arrived = all(src.iloc[-length - 1: -startpoint - 1] >= virtual_line_src) and \
                              all(close.iloc[-length - 1: -startpoint - 1] >= virtual_line_close)

                    if arrived:
                        divlen = length
                        break

        return divlen

    def negative_regular_negative_hidden_divergence(self, src: pd.Series, close: pd.Series,
                                                    ph_vals: np.array, ph_positions: np.array,
                                                    cond: int = 1):
        # function to check negative regular or positive hidden divergence
        # cond == 1 = > negative_regular, cond == 2 = > positive_hidden

        divlen = 0

        # if indicators less than last value or close price is less than last close
        if self.dont_wait_for_confirmation or src.iloc[-1] < src.iloc[-2] or close.iloc[-1] < close.iloc[-2]:
            startpoint = 0 if self.dont_wait_for_confirmation else 1

            for x in range(0, min(len(ph_positions), self.max_pivot_points)):
                length = src.index[-1] - ph_positions.iloc[-x - 1] + self.pivot_period
                # if we reach non valued array element or arrived 101. or previous bars then we don't search more
                if ph_positions.iloc[-x - 1] == 0 or length > self.max_bars_to_check - 1:
                    break
                if length > 5 and \
                        ((cond == 1 and src.iloc[-startpoint - 1] < src.iloc[-length - 1] and close.iloc[
                            -startpoint - 1] > ph_vals.iloc[-x - 1]) or \
                         (cond == 2 and src.iloc[-startpoint - 1] > src.iloc[-length - 1] and close.iloc[
                             -startpoint - 1] < ph_vals.iloc[-x - 1])):

                    virtual_line_src = np.linspace(src.iloc[-length - 1], src.iloc[-startpoint - 1],
                                                   length - startpoint)
                    virtual_line_close = np.linspace(close.iloc[-length - 1], close.iloc[-startpoint - 1],
                                                     length - startpoint)
                    arrived = all(src.iloc[-length - 1: -startpoint - 1] <= virtual_line_src) and \
                              all(close.iloc[-length - 1: -startpoint - 1] <= virtual_line_close)

                    if arrived:
                        divlen = length
                        break

        return divlen

    def calculate_divs(self, indicator: pd.Series, close: pd.Series,
                       pl_vals: np.array, pl_positions: np.array,
                       ph_vals: np.array, ph_positions: np.array) -> np.array:
        divs = np.zeros(4, dtype=int)

        if self.searchdiv in ["Regular", "Regular/Hidden"]:
            divs[0] = self.positive_regular_positive_hidden_divergence(indicator, close,
                                                                       pl_vals, pl_positions, 1)
            divs[1] = self.negative_regular_negative_hidden_divergence(indicator, close,
                                                                       ph_vals, ph_positions, 1)
        if self.searchdiv in ["Hidden", "Regular/Hidden"]:
            divs[2] = self.positive_regular_positive_hidden_divergence(indicator, close,
                                                                       pl_vals, pl_positions, 2)
            divs[3] = self.negative_regular_negative_hidden_divergence(indicator, close,
                                                                       ph_vals, ph_positions, 2)

        return divs

    def get_divergence_signal(self, data: pd.DataFrame) -> int:

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

        if div_signals["top"].sum() >= self.min_number_of_divergence:
            # if it was a pivot high after last top divergence
            if self.last_top_divergence is not None:

                idx = data.index.get_loc(self.last_top_divergence)
                if ph_positions.iloc[-1] <= idx:
                    return 0

            self.last_top_divergence = data.index[-1]
            return -1

        if div_signals["bottom"].sum() >= self.min_number_of_divergence:
            # if it was a pivot low after last bottom divergence
            if self.last_bottom_divergence is not None:

                idx = data.index.get_loc(self.last_bottom_divergence)
                if pl_positions.iloc[-1] <= idx:
                    return 0

            self.last_bottom_divergence = data.index[-1]
            return 1

        return 0

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
