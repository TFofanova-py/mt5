import datetime
from datetime import datetime
from time import sleep
import logging
import pandas as pd
import numpy as np
from .constants import MT5_TIMEFRAME
from typing import List, Union
from .external_history import get_yahoo_data, get_twelvedata
from .enums import DataSource, ConfigType, Direction
from .strategy import MultiIndStrategy, RelatedStrategy
from models.multiind_models import PairConfig, MT5Broker
from models.base_models import BasePairConfig, BaseOpenConfig, MakeTradingStepResponse, CheckedConfigResponse
from models.vix_models import RelatedPairConfig
import MetaTrader5Copy as mt2


class BasePair:
    def __init__(self, broker, pair_config: BasePairConfig):
        self.broker = broker
        self.strategy = None
        self.symbol = pair_config.symbol  # symbol of the instrument MT5
        self.datasource_symbol = pair_config.ds_symbol  # symbol of the instrument on datasource
        self.data_source = pair_config.data_source
        if self.data_source.name == DataSource.mt5:
            self.data_source.connection = MT5Broker.model_validate(self.data_source.connection)
            mt2.initialize(login=self.data_source.connection.login, password=self.data_source.connection.password,
                           server=self.data_source.connection.server, path=str(self.data_source.connection.path))
            print("Connection to the data source", self.symbol, mt2.last_error())
        if isinstance(pair_config.open_config, dict):
            self.open_config = BaseOpenConfig(**pair_config.open_config)
        else:
            self.open_config = pair_config.open_config
        self.close_config = pair_config.close_config
        self.deal_size = pair_config.deal_size  # volume for opening position must be a float
        self.devaition = pair_config.deviation  # max number of points to squeeze when open / close position
        # stop level coefficient for opening position
        self.broker_stop_coefficient = pair_config.broker_stop_coefficient

        mt5_symbol_info = self.broker.symbol_info(self.symbol)
        self.trade_tick_size = mt5_symbol_info.trade_tick_size if mt5_symbol_info is not None else 0.1
        self.positions: list = []
        self.orders: list = []
        self.last_check_time: dict = {ConfigType.open: None}
        if hasattr(self.close_config, "resolution"):
            self.last_check_time.update({ConfigType.close: None})

    @property
    def min_resolution(self):
        configs_with_resolution = [k for k in ConfigType if
                                   hasattr(self.__getattribute__(f"{k.value}_config"), "resolution")]
        return min([self.__getattribute__(f"{k.value}_config").resolution for k in configs_with_resolution])

    def get_historical_data(self, **kwargs):

        resolution = kwargs.get("resolution")
        numpoints = kwargs.get("numpoints")
        capital_conn = kwargs.get("capital_conn")

        if numpoints is None and hasattr(self.strategy, "numpoints"):
            numpoints = self.strategy.numpoints
        else:
            numpoints = 100

        if self.data_source.name == DataSource.mt5:

            try:
                if 60 <= resolution < 60 * 24 and resolution % 60 == 0:
                    interval = MT5_TIMEFRAME[str(resolution // 60) + "h"]
                elif resolution == 60 * 24:
                    interval = MT5_TIMEFRAME["1d"]
                elif resolution == 60 * 24 * 7:
                    interval = MT5_TIMEFRAME["1wk"]
                else:
                    interval = MT5_TIMEFRAME[str(resolution) + "m"]

            except KeyError:
                msg = f"{resolution} minutes is not a standard MetaTrade5 timeframe, choose another resolution"
                print(msg)
            else:
                rates = mt2.copy_rates_from_pos(self.datasource_symbol, interval, 0, numpoints)

                if rates is None:
                    print(mt2.last_error())
                    print(
                        f"{datetime.datetime.now().time().isoformat(timespec='minutes')} {self.symbol}: Can't get the historical data")
                    return None

                curr_prices = pd.DataFrame(rates)
                curr_prices["time"] = pd.to_datetime(curr_prices["time"], unit="s", utc=True)
                curr_prices.set_index("time", inplace=True)
                curr_prices = curr_prices[["close", "open", "high", "low", "tick_volume"]]
                curr_prices = curr_prices.rename({"tick_volume": "volume"}, axis=1)

                return curr_prices

        elif self.data_source.name == DataSource.capital:

            assert capital_conn is not None, "capital_conn mustn't be None"

            rates = capital_conn.get_capital_data(self.datasource_symbol, resolution, numpoints)

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

        elif self.data_source.name == DataSource.yahoo:
            interval: str = f"{resolution}m" if resolution < 60 else (
                f"{resolution // 60}h" if resolution < 24 * 60 else f"{resolution // (24 * 60)}d")
            assert interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d"], \
                f"{interval} is not valid for yfinance"
            assert self.datasource_symbol is not None, "datasource_symbol mustn't be None"

            y_data = get_yahoo_data(self.datasource_symbol, interval=interval, numpoints=numpoints)
            seconds_from_last_fixed_point = (y_data.index[-1] - y_data.index[-2]).seconds
            tol = 30  # in seconds
            if seconds_from_last_fixed_point < tol:
                return y_data.iloc[:-1]
            elif seconds_from_last_fixed_point < resolution * 60:
                sleep(resolution * 60 + tol // 3 - seconds_from_last_fixed_point)
                return get_yahoo_data(self.datasource_symbol, interval=interval, numpoints=numpoints).iloc[:-1]
            else:
                print(f"Data for {self.symbol}, {self.datasource_symbol} is weird")

        elif self.data_source.name == DataSource.twelvedata:
            return get_twelvedata(self.datasource_symbol, resolution, numpoints)

    def update_positions(self):
        self.positions = self.broker.positions_get(symbol=self.symbol)
        if self.positions is None:
            self.positions = []
            self.orders = []
        else:
            self.positions = [x for x in self.positions if x.identifier in self.orders]

    def get_configs_to_check(self) -> List[CheckedConfigResponse]:
        has_opened_positions = not (self.orders is None or len(self.orders) == 0)

        is_time_to_check = {
            k: True if v is None else (v - datetime.now()).seconds // 60 >= self.__getattribute__(
                f"{k}_config").resolution for k, v
            in self.last_check_time.items()}
        result = []
        if not has_opened_positions and is_time_to_check[ConfigType.open]:
            result.append(CheckedConfigResponse(applied_config=self.open_config,
                                                available_actions=[self.broker.ORDER_TYPE_BUY,
                                                                   self.broker.ORDER_TYPE_SELL]))
        if has_opened_positions and ConfigType.close in is_time_to_check and is_time_to_check[ConfigType.close]:
            result.append(CheckedConfigResponse(applied_config=self.close_config,
                                                available_actions=[self.broker.TRADE_ACTION_SLTP,
                                                                   (self.positions[0].type + 1) % 2]))
        return result

    def update_last_check_time(self, configs: List[CheckedConfigResponse]):
        for k in [cnf.applied_config.type for cnf in configs]:
            self.last_check_time[k] = datetime.now()

    def make_action(self, type_action: int, action_details: dict):
        print(f"{self.symbol}, Orders before making action: {self.orders}")

        curr_time = action_details["curr_time"]

        if type_action in [self.broker.ORDER_TYPE_BUY, self.broker.ORDER_TYPE_SELL]:
            assert "price" in action_details, f"{self.symbol}, Price is required for action type {type_action}"
            price = action_details["price"]
            positive_only = action_details.get("positive_only", False)

            if len(self.orders) == 0 or (
                    self.strategy.direction == "swing" and self.positions[0].type == type_action):
                response = self.create_position(price=price, type_action=type_action)
                if response and response.retcode == 10009:
                    self.orders.append(response.order)
                # self.resolution = self.close_config.resolution

                logging.info(f"{curr_time}, open position: {response}")
                print(response)

            else:
                responses = self.close_opened_position(price=price,
                                                       type_action=type_action,
                                                       positive_only=positive_only)
                print(responses)
                for resp in responses:
                    self.orders.remove(resp.order)
                logging.info(f"{curr_time}, close position: {responses}")

        elif type_action == self.broker.TRADE_ACTION_SLTP:
            # modify stop-loss
            assert "new_sls" in action_details, f"new_sls is required for modifying SL"
            new_sls = action_details["new_sls"]
            responses = self.modify_sl(new_sls)
            if any(responses):
                print("modify stop-loss", responses)
            logging.info(f"{curr_time}, modify position: {responses}")

        print(f"{self.symbol}, Orders after making action: {self.orders}")

    def make_trading_step(self) -> MakeTradingStepResponse:
        self.update_positions()

        configs_to_check = self.get_configs_to_check()
        self.update_last_check_time(configs_to_check)

        for cnf in configs_to_check:
            resolution = self.__getattribute__(f"{cnf.applied_config.type.value}_config").resolution
            data = self.get_historical_data(resolution=resolution)

            if data is None:
                return MakeTradingStepResponse(is_success=False, time_to_sleep=resolution * 60)

            type_action, action_details = self.strategy.get_action(data=data,
                                                                   symbol=self.symbol,
                                                                   positions=self.positions,
                                                                   stop_coefficient=self.broker_stop_coefficient,
                                                                   trade_tick_size=self.trade_tick_size,
                                                                   config=cnf.applied_config)

            if type_action in cnf.available_actions:
                self.make_action(type_action, action_details)
                sleep(5)
                self.update_positions()

            elif type_action is not None:
                print(
                    f"{datetime.now().time().isoformat(timespec='minutes')} {self.symbol}: Action {type_action} is not available for {cnf.applied_config.type.value} config and {len(self.positions)} positions and {self.strategy.direction} direction")

        time_to_sleep = min(
            [cnf.applied_config.resolution for cnf in configs_to_check],
            default=self.min_resolution)
        print(f"{datetime.now().time().isoformat(timespec='minutes')} {self.symbol}: Sleep for {time_to_sleep} minutes")
        return MakeTradingStepResponse(is_success=True, time_to_sleep=time_to_sleep)

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
            identifiers = self.orders

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


class MultiIndPair(BasePair):

    def __init__(self, broker, pair_config: PairConfig):
        super().__init__(broker, BasePairConfig.model_validate(pair_config.dict()))
        self.open_config = pair_config.open_config
        self.close_config = pair_config.close_config
        self.strategy = MultiIndStrategy(pair_config)

    def get_configs_to_check(self) -> List[CheckedConfigResponse]:
        result = super().get_configs_to_check()
        has_opened_positions = not (self.orders is None or len(self.orders) == 0)

        is_time_to_check = {
            k: True if v is None else (v - datetime.now()).seconds // 60 >= self.__getattribute__(
                f"{k}_config").resolution for k, v
            in self.last_check_time.items()}

        if has_opened_positions and self.strategy.direction == Direction.swing and is_time_to_check[ConfigType.open]:
            result.append(CheckedConfigResponse(applied_config=self.open_config,
                                                available_actions=[self.broker.TRADE_ACTION_SLTP,
                                                                   self.positions[0].type]))
        return result


class RelatedPair(BasePair):

    def __init__(self, broker, pair_config: RelatedPairConfig):
        super().__init__(broker, BasePairConfig.model_validate(pair_config.dict()))
        self.open_config = pair_config.open_config
        self.close_config = pair_config.close_config
        self.strategy = RelatedStrategy(pair_config)
