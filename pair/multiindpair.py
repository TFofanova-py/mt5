import datetime
from datetime import timedelta
from time import sleep
import logging
import pandas as pd
import numpy as np
from .constants import MT5_TIMEFRAME
from typing import Literal, List, Tuple
from .ta_utils import (
    rsi, macd, momentum, cci, obv,
    stk, vwmacd, cmf, mfi,
    pivotlow, pivothigh, bollinger_bands)
import os
from MetaTrader5 import OrderSendResult
from .external_history import get_yahoo_data, get_twelvedata


class MultiIndPair:

    def __init__(self, broker, strategy_id, kwargs):
        try:
            self.broker = broker
            self.strategy = MultiIndStrategy(strategy_id, **kwargs)
            self.symbol: str = kwargs["symbol"]  # symbol of the instrument MT5
            self.datasource_symbol: str = kwargs.get("datasource_symbol")  # symbol of the instrument on datasource
            self.data_source: Literal["mt5", "capital", "yahoo", "twelvedata"] = kwargs.get("data_source", "mt5")
            self.resolution: int = self.strategy.resolution_set["open"]
            self.min_resolution: int = min([v for v in self.strategy.resolution_set.values()])
            self.deal_size: float = float(kwargs.get("deal_size", 1.0))  # volume for opening position must be a float
            self.devaition: int = kwargs.get("deviation",
                                             50)  # max number of points to squeeze when open / close position
            # stop level coefficient for opening position
            self.stop_coefficient: float = kwargs.get("stop_coefficient", 0.9995)

            mt5_symbol_info = self.broker.symbol_info(self.symbol)
            self.trade_tick_size = mt5_symbol_info.trade_tick_size if mt5_symbol_info is not None else 0.1
            self.positions: list = []
            self.last_check_time: dict = {"open": None, "close": None}

        except TypeError as e:
            logging.error(f"Error: {e}")
            print(f"Error: {e}")

    def set_parameters_by_config(self, config: dict):
        key = config["applied_config"]
        self.resolution = self.strategy.resolution_set[key]
        self.strategy.pivot_period = self.strategy.pivot_period_set[key]
        self.strategy.indicators = self.strategy.indicators_set[key]

    def get_configs_to_check(self) -> List[dict]:
        has_opened_positions = not (self.positions is None or len(self.positions) == 0)
        is_time_to_check = {
            k: True if v is None else (v - datetime.datetime.now()).seconds // 60 >= self.strategy.resolution_set[k] for k, v
            in self.last_check_time.items()}
        result = []
        if not has_opened_positions and is_time_to_check["open"]:
            result.append({"applied_config": "open",
                           "available_actions": [self.broker.ORDER_TYPE_BUY, self.broker.ORDER_TYPE_SELL]})
        if has_opened_positions and is_time_to_check["close"]:
            result.append({"applied_config": "close",
                           "available_actions": [self.broker.TRADE_ACTION_SLTP, (self.positions[0].type + 1) % 2]})
        if has_opened_positions and self.strategy.direction == "swing" and is_time_to_check["open"]:
            result.append({"applied_config": "open",
                           "available_actions": [self.broker.TRADE_ACTION_SLTP, self.positions[0].type]})
        return result

    def update_last_check_time(self, configs: List[dict]):
        for k in [cnf["applied_config"] for cnf in configs]:
            self.last_check_time[k] = datetime.datetime.now()

    def update_positions(self):
        self.positions = self.broker.positions_get(symbol=self.symbol)
        if self.positions is None:
            self.positions = []

    def get_historical_data(self, **kwargs):

        numpoints = kwargs.get("numpoints", None)
        capital_conn = kwargs.get("capital_conn", None)

        if numpoints is None:
            numpoints = self.strategy.max_bars_to_check

        if self.data_source == "mt5":

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
                    print(f"{datetime.datetime.now().time().isoformat(timespec='minutes')} {self.symbol}: Can't get the historical data")
                    return None

                curr_prices = pd.DataFrame(rates)
                curr_prices["time"] = pd.to_datetime(curr_prices["time"], unit="s", utc=True)
                curr_prices.set_index("time", inplace=True)
                curr_prices = curr_prices[["close", "open", "high", "low", "tick_volume"]]
                curr_prices = curr_prices.rename({"tick_volume": "volume"}, axis=1)

                return curr_prices

        elif self.data_source == "capital":

            assert capital_conn is not None, "capital_conn mustn't be None"

            rates = capital_conn.get_capital_data(self.datasource_symbol, self.resolution, numpoints)

            if rates is None:
                print(f"{self.symbol}: Can't get the historical data")
                return None

            rates_agg = [{k: (v["bid"] + v["ask"]) / 2 if type(v) == dict else v for k, v in y.items()} for y in
                         rates]
            curr_prices = pd.DataFrame.from_dict(rates_agg, orient="columns")
            curr_prices = curr_prices.drop("snapshotTime", axis=1).set_index("snapshotTimeUTC")
            curr_prices = curr_prices.rename({"closePrice": "close", "highPrice": "high",
                                              "lowPrice": "low", "openPrice": "open",
                                              "lastTradedVolume": "volume"}, axis=1)
            return curr_prices

        elif self.data_source =="yahoo":
            interval: str = f"{self.resolution}m" if self.resolution < 60 else (f"{self.resolution//60}h" if self.resolution < 24 * 60 else f"{self.resolution//(24*60)}d")
            assert interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d"], \
                f"{interval} is not valid for yfinance"
            assert self.datasource_symbol is not None, "datasource_symbol mustn't be None"

            y_data = get_yahoo_data(self.datasource_symbol, interval=interval, numpoints=numpoints)
            seconds_from_last_fixed_point = (y_data.index[-1] - y_data.index[-2]).seconds
            tol = 30  # in seconds
            if seconds_from_last_fixed_point < tol:
                return y_data.iloc[:-1]
            elif seconds_from_last_fixed_point < self.resolution * 60:
                sleep(self.resolution * 60 + tol // 3 - seconds_from_last_fixed_point)
                return get_yahoo_data(self.datasource_symbol, interval=interval, numpoints=numpoints).iloc[:-1]
            else:
                print(f"Data for {self.symbol}, {self.datasource_symbol} is weird")

        elif self.data_source == "twelvedata":
            return get_twelvedata(self.datasource_symbol, self.resolution, numpoints)

    def create_position(self, price: float, type_action: int, sl: float = None):
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

    def close_opened_position(self, price: float, type_action: str, identifiers: List[int] = None,
                              positive_only: bool = False) -> list:
        if identifiers is None:
            identifiers = [pos.identifier for pos in self.broker.positions_get(symbol=self.symbol)]

        profits = [pos.profit for pos in self.broker.positions_get(symbol=self.symbol)
                       if pos.identifier in identifiers]

        responses = []
        for i, position in enumerate(identifiers):
            if not positive_only or profits[i] > 0:
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
            else:
                responses.append(f"Don't close negative position {position}")

        return responses

    def modify_sl(self, new_sls: List[float], identifiers: List[int] = None) -> list:
        if identifiers is None:
            identifiers = [pos.identifier for pos in self.broker.positions_get(symbol=self.symbol)]

        assert new_sls is not None
        assert len(identifiers) == len(new_sls)

        responses = []
        for position, new_sl in zip(identifiers, new_sls):
            if new_sl:
                request = {
                    "action": self.broker.TRADE_ACTION_SLTP,
                    "symbol": self.symbol,
                    "sl": new_sl,
                    "position": position
                }
                # check order before placement
                # check_result = broker.order_check(request)
                responses.append(self.broker.order_send(request))
            else:
                responses.append(None)

        return responses

    def make_action(self, type_action: int, action_details: dict):

        curr_time = action_details["curr_time"]

        if type_action in [self.broker.ORDER_TYPE_BUY, self.broker.ORDER_TYPE_SELL]:
            assert "price" in action_details, f"{self.symbol}, Price is required for action type {type_action}"
            price = action_details["price"]
            positive_only = action_details.get("positive_only", False)

            if len(self.positions) == 0 or (
                    self.strategy.direction == "swing" and self.positions[0].type == type_action):
                response = self.create_position(price=price, type_action=type_action)
                self.resolution = self.strategy.resolution_set["close"]

                logging.info(f"{curr_time}, open position: {response}")
                print(response)

            else:
                responses = self.close_opened_position(price=price, type_action=type_action, positive_only=positive_only)
                print(responses)
                logging.info(f"{curr_time}, close position: {responses}")
                # self.save_trade(responses, file_name="../trades.csv")

        elif type_action == self.broker.TRADE_ACTION_SLTP:
            # modify stop-loss
            assert "new_sls" in action_details, f"new_sls is required for modifying SL"
            new_sls = action_details["new_sls"]
            responses = self.modify_sl(new_sls)
            if any(responses):
                print("modify stop-loss", responses)
            logging.info(f"{curr_time}, modify position: {responses}")

    def was_stoploss(self):
        to_dt = datetime.datetime.now()
        from_dt = to_dt - timedelta(minutes=self.resolution)
        self.broker.HistorySelect(from_dt, to_dt)
        total = self.broker.HistoryDealsTotal()
        for i in range(total):
            ticket = self.broker.HistoryDealGetTicket(i)
            if self.broker.HistoryDealGetInteger(ticket, self.broker.DEAL_REASON) == self.broker.DEAL_REASON_SL:
                sl_time = self.broker.HistoryDealGetInteger(ticket, self.broker.DEAL_TIME)
                sl_price = self.broker.HistoryDealGetDouble(ticket, self.broker.DEAL_PRICE)
                return True, (sl_time, sl_price)

        return False, ()

    def save_trade(self,
                   reason: Literal["Normal", "Stoploss"],
                   responses: List[OrderSendResult],
                   stoploss_details: Tuple[datetime.datetime, float],
                   file_name: str = "../trades.csv"):
        df_trade = None
        if os.path.exists(file_name):
            df_trade = pd.read_csv(file_name)

        data = None

        if reason == "Normal":
            response = responses[0]
            id_open_position = response.request.position

            if not self.broker.HistorySelectByPosition(id_open_position):
                logging.error("Something wrong with getting info about opening position, can't save trade")

            total = self.broker.HistoryDealsTotal()

            start_ticket = self.broker.HistoryDealGetTicket(total - 2)
            end_ticket = self.broker.HistoryDealGetTicket(total - 1)

            if start_ticket and end_ticket:
                start_time = self.broker.HistoryDealGetInteger(start_ticket, self.broker.DEAL_TIME)
                start_price = self.broker.HistoryDealGetDouble(start_ticket, self.broker.DEAL_PRICE)
                end_time = self.broker.HistoryDealGetInteger(end_ticket, self.broker.DEAL_TIME)
                end_price = self.broker.HistoryDealGetDouble(end_ticket, self.broker.DEAL_PRICE)

                data = pd.DataFrame.from_dict({"time_start": [start_time],
                                               "time_end": [end_time],
                                               "symbol": [self.symbol],
                                               "type_id": [0 if response.type == 1 else 1],  # 0 - long, 1 - short
                                               "ds_id": [0 if self.data_source == "mt5" else 1],
                                               # 0 - mt5, 1 - capital.com
                                               "strategy_id": [self.strategy.id],
                                               "size": [response.request.volume],
                                               "deviation": [response.request.deviation],
                                               "price_start": [start_price],
                                               "price_end": [end_price],
                                               "reason_end": [reason]})
        else:
            pass

        if data is not None:
            if df_trade is None:
                data["id"] = 0
                df_trade = data

            else:
                data["id"] = df_trade["id"].max() + 1
                df_trade = pd.concat([df_trade, data])

            df_trade.to_csv(file_name, index=False)


class MultiIndStrategy:
    def __init__(self, strategy_id: int, **kwargs):
        self.id: int = strategy_id
        self.resolution_set: dict = {"open": kwargs.get("open_config", {}).get("resolution", 3),
                                     "close": kwargs.get("close_config", {}).get("resolution", 3)}
        self.pivot_period_set: dict = {"open": kwargs.get("open_config", {}).get("pivot_period", 5),
                                       "close": kwargs.get("close_config", {}).get("pivot_period", 5)}
        self.pivot_period: int = self.pivot_period_set["open"]

        self.divtype: Literal["Regular", "Regular/Hidden", "Hidden"] = kwargs.get("divergence_type", "Regular")
        self.min_number_of_divergence: dict = kwargs.get("min_number_of_divergence",
                                                         {"entry": 1,
                                                          "exit_sl": 1,
                                                          "exit_tp": 1})
        self.max_pivot_points: int = kwargs.get("max_pivot_points", 10)
        self.max_bars_to_check: int = kwargs.get("max_bars_to_check", 100)
        self.dont_wait_for_confirmation: bool = kwargs.get("dont_wait_for_confirmation", True)
        self.indicators_set: dict = {"open": kwargs.get("open_config", {}).get("entry", {}),
                                     "close": kwargs.get("close_config", {}).get("entry", {})}
        self.indicators = self.indicators_set["open"]
        self.direction: Literal["low-long", "high-short", "bi", "swing"] = kwargs.get("direction", "low-long")
        self.entry_price_higher_than = kwargs.get("entry_price_higher_than")
        self.entry_price_lower_than = kwargs.get("entry_price_lower_than")
        self.exit_target = kwargs.get("exit_target")
        self.last_divergence = {"open": {"top": None, "bottom": None}, "close": {"top": None, "bottom": None}}
        self.bollinger = kwargs.get("open_config", {}).get("bollinger")
        self.close_positive_only: bool = kwargs.get("close_config", {}).get("positive_only", False)
        self.next_position_bol_check: bool = kwargs.get("open_config", {}).get("next_position_bol_check", True)

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

        if self.divtype in ["Regular", "Regular/Hidden"]:
            divs[0] = self.divergence_length(indicator, close, pl_vals, pl_positions, "positive_regular")
            divs[1] = self.divergence_length(indicator, close, ph_vals, ph_positions, "negative_regular")

        if self.divtype in ["Hidden", "Regular/Hidden"]:
            divs[2] = self.divergence_length(indicator, close, pl_vals, pl_positions, "positive_hidden")
            divs[3] = self.divergence_length(indicator, close, ph_vals, ph_positions, "negative_hidden")

        return divs

    def count_divergence(self, data: pd.DataFrame, config_type: Literal["open", "close"]) -> dict:

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
                if self.last_divergence[config_type][t] is not None and self.last_divergence[config_type][t] >= data.index[0]:
                    idx = data.index.get_loc(self.last_divergence[config_type][t])

                    new_divergence = (pl_positions.iloc[-1] > idx) if t == "bottom" else (ph_positions.iloc[-1] > idx)

                if new_divergence:
                    self.last_divergence[config_type][t] = data.index[-1]
                    triggered_idx = np.any(div_signals > 0, axis=1).tolist()
                    triggered_inds = [idx_name for idx, idx_name in enumerate(indices) if triggered_idx[idx]]

                    return {"top": div_signals["top"].sum(),
                            "bottom": div_signals["bottom"].sum(), "triggered": triggered_inds}

        return {"top": 0, "bottom": 0, "triggered": []}

    @staticmethod
    def was_price_goes_up(data: pd.DataFrame) -> bool:
        assert "close" in data.columns
        assert data.shape[0] > 1
        if data["close"].iloc[-1] > data["close"].iloc[-2]:
            return True
        return False

    def get_bollinger_conditions(self, data: pd.DataFrame) -> Tuple[bool, bool]:
        price = data["close"].iloc[-1]
        if self.bollinger is None:
            return True, True
        params = self.bollinger
        ind_bollinger = bollinger_bands(data, **params).iloc[-1][["upper", "lower"]]
        return price < ind_bollinger["lower"], price > ind_bollinger["upper"]

    def get_action(self, data: pd.DataFrame, pair: MultiIndPair, config_type: Literal["open", "close"]) -> Tuple[int, dict]:
        divergences_cnt = self.count_divergence(data, config_type=config_type)

        type_action = None
        details = {"curr_time": data.index[-1]}
        price = data["close"].iloc[-1]

        bollinger_cond_lower, bollinger_cond_upper = self.get_bollinger_conditions(data)

        if divergences_cnt["top"] + divergences_cnt["bottom"] > 0:
            print(f"{datetime.datetime.now().time().isoformat(timespec='minutes')} {pair.symbol}: divergences {divergences_cnt}, bollinger_lower: {bollinger_cond_lower}, bollinger_upper: {bollinger_cond_upper}")
            logging.info(f"{data.index[-1]}, divergences: {divergences_cnt}, bollinger_lower: {bollinger_cond_lower}, bollinger_upper: {bollinger_cond_upper}")

        if self.direction in ["low-long", "bi", "swing"] and \
                divergences_cnt["bottom"] >= self.min_number_of_divergence["entry"] and \
                bollinger_cond_lower and \
                len(pair.positions) == 0 and \
                (self.direction != "low-long" or
                 self.entry_price_lower_than is None or
                 (self.entry_price_lower_than and price < self.entry_price_lower_than)):
            type_action = pair.broker.ORDER_TYPE_BUY

        elif self.direction in ["high-short", "bi", "swing"] and \
                divergences_cnt["top"] >= self.min_number_of_divergence["entry"] and \
                bollinger_cond_upper and \
                len(pair.positions) == 0 and \
                (self.direction != "high-short" or
                 self.entry_price_higher_than is None or
                 (self.entry_price_higher_than and price > self.entry_price_higher_than)):
            type_action = pair.broker.ORDER_TYPE_SELL

        elif len(pair.positions) > 0:
            same_direction_divergences = divergences_cnt["bottom"] if pair.positions[0].type == 0 \
                else divergences_cnt["top"]
            opposite_direction_divergences = divergences_cnt["top"] if pair.positions[0].type == 0 else divergences_cnt[
                "bottom"]

            print(datetime.datetime.now().time().isoformat(timespec='minutes'), pair.symbol, self.direction, "len pos", len(pair.positions),
                  "same direction", same_direction_divergences, "opposite direction", opposite_direction_divergences)

            if ((self.direction != "swing" and same_direction_divergences >= self.min_number_of_divergence["exit_sl"]) or \
                    opposite_direction_divergences >= self.min_number_of_divergence["exit_tp"]) and \
                    (self.exit_target is None or (self.direction == "high-short") * (price < self.exit_target)
                     or (self.direction == "low-long") * (price > self.exit_target)):
                # close position
                type_action = (pair.positions[0].type + 1) % 2
                details.update({"positive_only": self.close_positive_only})

            elif self.direction == "swing" and same_direction_divergences >= self.min_number_of_divergence["entry"]:
                bollinger_cond = None
                if self.next_position_bol_check:
                    bollinger_cond = (pair.positions[0].type == 0 and bollinger_cond_lower) or (
                            pair.positions[0].type == 1 and bollinger_cond_upper)
                    print(f"{datetime.datetime.now().time().isoformat(timespec='minutes')} {pair.symbol}: bollinger common: {bollinger_cond}")

                if not self.next_position_bol_check or bollinger_cond:
                    # open another position
                    type_action = pair.positions[0].type
                    print(
                        f"{datetime.datetime.now().time().isoformat(timespec='minutes')} {pair.symbol}: type_action: {type_action}")
            else:
                # modify stop-loss
                price_goes_up = self.was_price_goes_up(data)
                new_sls = None

                if pair.positions[0].type == 0 and price_goes_up:
                    # if long and price goes up, move sl up
                    new_sl = round(price * pair.stop_coefficient, abs(int(np.log10(pair.trade_tick_size))))
                    new_sls = [new_sl if x.sl < new_sl else None for x in pair.positions]
                elif pair.positions[0].type == 1 and not price_goes_up:
                    # if short and price goes down, move sl down
                    new_sl = round(price * (2. - pair.stop_coefficient), abs(int(np.log10(pair.trade_tick_size))))
                    new_sls = [new_sl if x.sl > new_sl else None for x in pair.positions]

                if new_sls is not None:
                    type_action = pair.broker.TRADE_ACTION_SLTP
                    details.update({"new_sls": new_sls})

        if type_action in [pair.broker.ORDER_TYPE_BUY, pair.broker.ORDER_TYPE_SELL]:
            details.update({"price": data["close"].iloc[-1]})

        return type_action, details
