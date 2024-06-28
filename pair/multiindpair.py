import datetime
from datetime import timedelta
from time import sleep
import logging
import pandas as pd
import numpy as np
from .constants import MT5_TIMEFRAME
from typing import List, Union
from .external_history import get_yahoo_data, get_twelvedata
from .enums import DataSource, ConfigType
from .strategy import MultiIndStrategy
from models import PairConfig


class MultiIndPair:

    def __init__(self, broker, pair_config: PairConfig):
        try:
            self.broker = broker
            self.strategy = MultiIndStrategy(pair_config)
            self.symbol = pair_config.symbol  # symbol of the instrument MT5
            self.datasource_symbol = pair_config.ds_symbol  # symbol of the instrument on datasource
            self.data_source = pair_config.data_source
            self.open_config = pair_config.open_config
            self.close_config = pair_config.close_config
            self.resolution = self.open_config.resolution
            self.min_resolution = min([self.__getattribute__(f"{k.value}_config").resolution for k in ConfigType])
            self.deal_size = pair_config.deal_size  # volume for opening position must be a float
            self.devaition = pair_config.deviation  # max number of points to squeeze when open / close position
            # stop level coefficient for opening position
            self.broker_stop_coefficient = pair_config.broker_stop_coefficient

            mt5_symbol_info = self.broker.symbol_info(self.symbol)
            self.trade_tick_size = mt5_symbol_info.trade_tick_size if mt5_symbol_info is not None else 0.1
            self.positions: list = []
            self.last_check_time: dict = {ConfigType.open: None, ConfigType.close: None}

        except TypeError as e:
            logging.error(f"Error: {e}")
            print(f"Error: {e}")

    def set_parameters_by_config(self, config: dict):
        key = config["applied_config"]
        self.resolution = self.__getattribute__(f"{key}_config").resolution
        self.strategy.pivot_period = self.__getattribute__(f"{key}_config").pivot_period
        self.strategy.indicators = self.__getattribute__(f"{key}_config").entry

    def get_configs_to_check(self) -> List[dict]:
        has_opened_positions = not (self.positions is None or len(self.positions) == 0)

        is_time_to_check = {
            k: True if v is None else (v - datetime.datetime.now()).seconds // 60 >= self.__getattribute__(f"{k}_config").resolution for k, v
            in self.last_check_time.items()}
        result = []
        if not has_opened_positions and is_time_to_check[ConfigType.open]:
            result.append({"applied_config": ConfigType.open.value,
                           "available_actions": [self.broker.ORDER_TYPE_BUY, self.broker.ORDER_TYPE_SELL]})
        if has_opened_positions and is_time_to_check[ConfigType.close]:
            result.append({"applied_config": ConfigType.close.value,
                           "available_actions": [self.broker.TRADE_ACTION_SLTP, (self.positions[0].type + 1) % 2]})
        if has_opened_positions and self.strategy.direction == "swing" and is_time_to_check[ConfigType.open]:
            result.append({"applied_config": ConfigType.open.value,
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

        if self.data_source == DataSource.mt5:

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
                    print(self.broker.last_error())
                    print(f"{datetime.datetime.now().time().isoformat(timespec='minutes')} {self.symbol}: Can't get the historical data")
                    return None

                curr_prices = pd.DataFrame(rates)
                curr_prices["time"] = pd.to_datetime(curr_prices["time"], unit="s", utc=True)
                curr_prices.set_index("time", inplace=True)
                curr_prices = curr_prices[["close", "open", "high", "low", "tick_volume"]]
                curr_prices = curr_prices.rename({"tick_volume": "volume"}, axis=1)

                return curr_prices

        elif self.data_source == DataSource.capital:

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

        elif self.data_source == DataSource.yahoo:
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

        elif self.data_source == DataSource.twelvedata:
            return get_twelvedata(self.datasource_symbol, self.resolution, numpoints)

    def create_position(self, price: float, type_action: int, sl: float = None):
        if sl is None:
            if type_action == 0:
                sl = round(price * self.broker_stop_coefficient, abs(int(np.log10(self.trade_tick_size))))
            else:
                sl = round(price * (2. - self.broker_stop_coefficient), abs(int(np.log10(self.trade_tick_size))))

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

    def close_opened_position(self, price: float, type_action: Union[str, int], identifiers: List[int] = None,
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
                self.resolution = self.close_config.resolution

                logging.info(f"{curr_time}, open position: {response}")
                print(response)

            else:
                responses = self.close_opened_position(price=price,
                                                       type_action=type_action,
                                                       positive_only=positive_only)
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
