import datetime
from datetime import datetime, time
from time import sleep
import logging
import pandas as pd
import numpy as np
from .constants import MT5_TIMEFRAME
from typing import List, Union
from .external_history import get_yahoo_data, get_twelvedata
from .enums import DataSource, ConfigType, Direction, ActionMethod
from .strategy import MultiIndStrategy, RelatedStrategy
from models.multiind_models import PairConfig
from models.base_models import BasePairConfig, BaseOpenConfig, MakeTradingStepResponse, CheckedConfigResponse, MT5Broker, ActionDetails
from models.vix_models import RelatedPairConfig, RelatedOpenConfig
import MetaTrader5Copy as mt2
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


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
            print("Connection to the data source", self.datasource_symbol, mt2.last_error())
        self.action_methods = pair_config.action_methods
        self.alarm_config = pair_config.alarm_config if ActionMethod.alarm in pair_config.action_methods and pair_config.alarm_config else None
        if isinstance(pair_config.open_config, dict):
            self.open_config = BaseOpenConfig(**pair_config.open_config)
        else:
            self.open_config = pair_config.open_config
        self.close_config = pair_config.close_config
        self.deal_size = pair_config.deal_size  # volume for opening position must be a float
        self.devaition = pair_config.deviation  # max number of points to squeeze when open / close position
        self.time_to_trade = pair_config.time_to_trade
        # stop level coefficient for opening position
        self.broker_stop_coefficient = pair_config.broker_stop_coefficient
        self.broker_take_profit = pair_config.broker_take_profit

        mt5_symbol_info = self.broker.symbol_info(self.symbol)
        self.trade_tick_size = mt5_symbol_info.trade_tick_size if mt5_symbol_info is not None else 0.1
        self.positions: list = []
        self.orders: list = []
        self.last_check_time: dict = {ConfigType.open: None}
        if self.close_config.get("resolution"):
            self.last_check_time.update({ConfigType.close: None})
        self.logger = logging.getLogger(__name__)

    @property
    def min_resolution(self):
        configs_with_resolution = [k for k in ConfigType if
                                   hasattr(self.__getattribute__(f"{k.value}_config"), "resolution")]
        return min([self.__getattribute__(f"{k.value}_config").resolution for k in configs_with_resolution])

    @staticmethod
    def get_seconds_to_time(t: time) -> int:
        t_now = datetime.now().time()
        if t_now >= t:
            return 0

        return (((t.hour - t_now.hour + 24) % 24) * 60 * 60 +
                (((t.minute - t_now.minute + 60) % 60) * 60) +
                (t.second - t_now.second + 60) % 60)

    def get_time_to_sleep(self, configs: List[CheckedConfigResponse]) -> float:
       return min(
            [cnf.applied_config.resolution for cnf in configs],
            default=self.min_resolution)

    def get_historical_data(self, **kwargs):

        resolution = kwargs.get("resolution")
        numpoints = kwargs.get("numpoints")
        capital_conn = kwargs.get("capital_conn")
        start_date = kwargs.get("start_date")

        if numpoints is None and hasattr(self.strategy, "numpoints"):
            numpoints = self.strategy.numpoints
        elif numpoints is None:
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
                        f"{datetime.now().time().isoformat(timespec='minutes')} {self.symbol}: Can't get the historical data")
                    return None

                curr_prices = pd.DataFrame(rates)
                curr_prices["time"] = pd.to_datetime(curr_prices["time"], unit="s", utc=True)
                curr_prices.set_index("time", inplace=True)
                if start_date:
                    filtered = curr_prices.index[curr_prices.index < start_date.strftime('%Y-%m-%d %H:%M:%S')]
                    start_loc = curr_prices.index.get_loc(filtered[-1]) if len(filtered) > 0 else 0
                    curr_prices = curr_prices.iloc[start_loc - (self.strategy.numpoints or 100):]
                curr_prices = curr_prices[["close", "open", "high", "low", "tick_volume"]]
                curr_prices = curr_prices.rename({"tick_volume": "volume"}, axis=1)

                return curr_prices.resample(f"{resolution}Min").bfill()

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
            if start_date:
                start_loc = curr_prices.index.get_loc(start_date)
                curr_prices = curr_prices.iloc[-(start_loc + numpoints):]
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

    def get_configs_to_check(self, curr_time: datetime = None) -> List[CheckedConfigResponse]:
        has_opened_positions = not (self.orders is None or len(self.orders) == 0)

        if not curr_time:
            curr_time = datetime.now()

        is_time_to_check = {
            k: True if v is None else (v - curr_time).seconds // 60 >= self.__getattribute__(
                f"{k}_config").resolution for k, v
            in self.last_check_time.items()}
        result = []
        if is_time_to_check[ConfigType.open] and curr_time.hour not in self.open_config.no_entry_hours:  # and not has_opened_positions
            result.append(CheckedConfigResponse(applied_config=self.open_config,
                                                available_actions=[self.broker.ORDER_TYPE_BUY,
                                                                   self.broker.ORDER_TYPE_SELL]))
        if has_opened_positions and ConfigType.close in is_time_to_check and is_time_to_check[ConfigType.close]:
            result.append(CheckedConfigResponse(applied_config=self.close_config,
                                                available_actions=[self.broker.TRADE_ACTION_SLTP,
                                                                   (self.positions[0].type + 1) % 2]))
        return result

    def update_last_check_time(self, configs: List[CheckedConfigResponse], curr_time: datetime = None):
        if not curr_time:
            curr_time = datetime.now()

        for k in [cnf.applied_config.type for cnf in configs]:
            self.last_check_time[k] = curr_time

        if self.last_check_time[ConfigType.close.value] is None:
            self.last_check_time[ConfigType.close.value] = curr_time

    def make_trade_action(self, type_action: int, action_details: ActionDetails, **kwargs):
        print(f"{self.symbol}, Orders before making action: {self.orders}")

        curr_time = action_details.curr_time

        if type_action in [self.broker.ORDER_TYPE_BUY, self.broker.ORDER_TYPE_SELL]:
            positive_only = action_details.positive_only

            if len(self.orders) == 0 or self.positions[0].type == type_action:
                response = self.create_position(price=action_details.price,
                                                type_action=type_action,
                                                deal_size=action_details.deal_size,
                                                **kwargs)
                # if order is completed
                if response and response.retcode == 10009:
                    self.orders.append(response.order)

                logging.info(f"{curr_time}, open position: {response}")
                print(response)

            else:
                responses = self.close_opened_position(price=action_details.price,
                                                       type_action=type_action,
                                                       positive_only=positive_only,
                                                       **kwargs)
                print(responses)
                for resp in responses:
                    if resp and not isinstance(resp, str):
                        if resp.retcode == 10009:
                            self.orders.remove(resp.request.position)
                logging.info(f"{curr_time}, close position: {responses}")

        elif type_action == self.broker.TRADE_ACTION_SLTP:
            # modify stop-loss
            new_sls = action_details.new_sls
            responses = self.modify_sl(new_sls=new_sls, identifiers=action_details.identifiers, type_action=type_action, **kwargs)
            if any(responses):
                print("modify stop-loss", responses)
            logging.info(f"{curr_time}, modify position: {responses}")

        print(f"{self.symbol}, Orders after making action: {self.orders}")

    def send_email(self, recipient_email, subject, body):
        try:
            # Create the email
            message = MIMEMultipart()
            message["From"] = self.alarm_config.parameters.user
            message["To"] = recipient_email
            message["Subject"] = subject

            # Add body to email
            message.attach(MIMEText(body, "plain"))

            # Connect to the SMTP server
            with smtplib.SMTP(self.alarm_config.parameters.server, self.alarm_config.parameters.port) as server:
                if self.alarm_config.parameters.connection_security == "STARTTLS":
                    server.starttls()
                server.login(self.alarm_config.parameters.user, self.alarm_config.parameters.password)
                server.sendmail(self.alarm_config.parameters.user, recipient_email, message.as_string())

            print("Email sent successfully!")
        except Exception as e:
            print(f"Error: {e}")

    def make_alarm_action(self, type_action: int):
        if type_action in [0, 1]:
            for recipient in self.alarm_config.recipients:
                self.send_email(recipient,
                                "Signal",
                                f"{datetime.now()}: {self.symbol} - {'BUY' if type_action == 0 else 'SELL'}")

    def make_action(self, type_action: int, action_details: ActionDetails, **kwargs):
        for method in self.action_methods:
            if method == ActionMethod.trade:
                self.make_trade_action(type_action, action_details, **kwargs)
            elif method == ActionMethod.alarm:
                self.make_alarm_action(type_action)

    def make_trading_step(self) -> MakeTradingStepResponse:
        self.update_positions()

        configs_to_check = self.get_configs_to_check()
        self.update_last_check_time(configs_to_check)

        for cnf in configs_to_check:
            resolution = cnf.applied_config.resolution
            data = self.get_historical_data(resolution=resolution)

            if data is None:
                return MakeTradingStepResponse(is_success=False, time_to_sleep=resolution * 60)

            type_action, action_details = self.strategy.get_action(data=data,
                                                                   symbol=self.symbol,
                                                                   positions=self.positions,
                                                                   stop_coefficient=self.broker_stop_coefficient,
                                                                   trade_tick_size=self.trade_tick_size,
                                                                   config=cnf.applied_config,
                                                                   verbose=True)
            self.logger.info(f"{datetime.now()}, action: {type_action} {action_details if type_action else ''}")

            if type_action in cnf.available_actions:

                if len(self.orders) == 0 and self.time_to_trade:
                    print(f"Waiting for {self.time_to_trade} to trade")
                    sleep(self.get_seconds_to_time(self.time_to_trade))

                kwargs = {}
                if hasattr(self.strategy, "bot_stop_coefficient"):
                    kwargs = {"bot_stop_coefficient": self.strategy.bot_stop_coefficient}

                self.make_action(type_action, action_details, **kwargs)
                sleep(5)
                self.update_positions()

            elif type_action is not None:
                print(
                    f"{datetime.now().time().isoformat(timespec='minutes')} {self.symbol}: Action {type_action} is not available for {cnf.applied_config.type.value} config and {len(self.positions)} positions and {self.strategy.direction} direction")

        time_to_sleep = self.get_time_to_sleep(configs=configs_to_check)
        print(f"{datetime.now().time().isoformat(timespec='minutes')} {self.datasource_symbol}: Sleep for {time_to_sleep} minutes")
        return MakeTradingStepResponse(is_success=True, time_to_sleep=time_to_sleep)

    def create_position(self, price: float, type_action: int, sl: float = None, deal_size: float = None, **kwargs):
        volume = deal_size if deal_size else self.deal_size

        if sl is None:
            if type_action == 0:
                sl = round(price * self.broker_stop_coefficient, abs(int(np.log10(self.trade_tick_size))))
            else:
                sl = round(price * (2. - self.broker_stop_coefficient), abs(int(np.log10(self.trade_tick_size))))

        request = {
            "action": self.broker.TRADE_ACTION_DEAL,  # for non market order TRADE_ACTION_PENDING
            "symbol": self.symbol,
            "volume": volume,
            "deviation": self.devaition,
            "type": type_action,
            "price": price,
            "sl": sl,
            "comment": "python script open",
            "type_time": self.broker.ORDER_TIME_GTC,
            "type_filling": self.broker.ORDER_FILLING_IOC
        }

        if self.broker_take_profit:
            request.update({"tp": round(self.broker_take_profit * price, abs(int(np.log10(self.trade_tick_size))))})

        # check order before placement
        check_result = self.broker.order_check(request)

        # if the order is incorrect
        if check_result.retcode != 0:
            # error codes are here: https://www.mql5.com/en/docs/constants/errorswarnings/enum_trade_return_codes
            print(check_result.retcode, check_result.comment, request)
            return None

        return self.broker.order_send(request)

    def close_opened_position(self, price: float, type_action: Union[str, int], identifiers: List[int] = None,
                              positive_only: bool = False, **kwargs) -> list:
        if identifiers is None:
            identifiers = self.orders

        bot_stop_coefficient = kwargs.get("bot_stop_coefficient")

        positions = [pos for pos in self.broker.positions_get(symbol=self.symbol)
                   if pos.identifier in identifiers]

        responses = []
        for i, position in enumerate(positions):
            request = {
                "action": self.broker.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": self.deal_size,
                "deviation": self.devaition,
                "type": type_action,
                "position": position.identifier,
                "price": price,
                "comment": "python script close",
                "type_time": self.broker.ORDER_TIME_GTC,
                "type_filling": self.broker.ORDER_FILLING_IOC
            }

            func_stop = np.less if position.type == 0 else lambda x, y: np.greater(x, 2 - y)
            if func_stop(position.price_current / position.price_open, bot_stop_coefficient):
                print(f'{position.symbol}: Close position {position.identifier} because of bot stop {bot_stop_coefficient}')
                responses.append(self.broker.order_send(request))

            elif not positive_only or position.profit >= 0:
                responses.append(self.broker.order_send(request))

            elif positive_only and position.profit < 0:
                responses.append(f"Don't close negative position {position}")

        return responses

    def modify_sl(self, new_sls: List[float], identifiers: List[int] = None, **kwargs) -> list:
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

    def get_configs_to_check(self, curr_time: datetime = None) -> List[CheckedConfigResponse]:
        if not curr_time:
            curr_time = datetime.now()

        result = super().get_configs_to_check(curr_time=curr_time)
        has_opened_positions = not (self.orders is None or len(self.orders) == 0)

        is_time_to_check = {
            k: True if v is None else (v - curr_time).seconds // 60 >= self.__getattribute__(
                f"{k}_config").resolution for k, v
            in self.last_check_time.items()}

        if has_opened_positions and self.strategy.direction in [Direction.swing, Direction.low_long, Direction.high_short] and is_time_to_check[ConfigType.open]:
            result.append(CheckedConfigResponse(applied_config=self.open_config,
                                                available_actions=[self.broker.TRADE_ACTION_SLTP,
                                                                   self.positions[0].type]))
        return result


class RelatedPair(BasePair):

    def __init__(self, broker, pair_config: RelatedPairConfig):
        pair_config.open_config.resolution = pair_config.open_config.candle_minutes
        pair_config.open_config.rebuy_config.resolution = pair_config.open_config.rebuy_config.check_for_every_minutes
        combined_data = {**pair_config.dict(),
                         "symbol": pair_config.ticker_to_trade,
                         "ds_symbol": pair_config.ticker_to_monitor,
                         "open_config": pair_config.open_config.dict()}
        super().__init__(broker, BasePairConfig.model_validate(combined_data))
        self.open_config = RelatedOpenConfig.model_validate(pair_config.open_config.dict())
        self.close_config = pair_config.close_config
        self.strategy = RelatedStrategy()

    def get_configs_to_check(self) -> List[CheckedConfigResponse]:
        result = super().get_configs_to_check()
        has_opened_positions = not (self.orders is None or len(self.orders) == 0)

        is_time_to_check = {
            k: True if v is None else (v - datetime.now()).seconds // 60 >= self.__getattribute__(
                f"{k}_config").resolution for k, v
            in self.last_check_time.items()}

        if has_opened_positions and self.open_config.rebuy_config.is_allowed and is_time_to_check[ConfigType.open]:
            result.append(CheckedConfigResponse(applied_config=self.open_config.rebuy_config,
                                                available_actions=[self.positions[0].type]))
        return result

    @property
    def min_resolution(self):
        has_opened_positions = not (self.orders is None or len(self.orders) == 0)
        if not has_opened_positions or not self.open_config.rebuy_config.is_allowed:
            return self.open_config.resolution
        return min(self.open_config.resolution, self.open_config.rebuy_config.resolution)

    def get_time_to_sleep(self, configs: List[CheckedConfigResponse]) -> float:
        if len(self.positions) == 0 and self.open_config.time_to_monitor:
            time_to_monitor = self.get_seconds_to_time(self.open_config.time_to_monitor) // 60
            return time_to_monitor if time_to_monitor > 0 else self.open_config.resolution

        return min(
            [cnf.applied_config.resolution for cnf in configs],
            default=self.min_resolution)
