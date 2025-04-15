import datetime
from datetime import time
from pydantic import BaseModel, FilePath
from pair.enums import DataSource, Direction, ConfigType, ActionMethod
from typing import Union, List, Any


class Symbol(BaseModel):
    ds_symbol: str
    deal_size: float = 1.0
    direction: Direction = Direction.low_long


class MT5Broker(BaseModel):
    login: int
    password: str
    server: str
    path: FilePath


class DataSourceBroker(BaseModel):
    name: DataSource = DataSource.mt5
    connection: Union[MT5Broker, dict, None] = None


class EmailParameters(BaseModel):
    server: str
    port: int
    auth_method: str = "normal password"
    connection_security: str = "STARTTLS"
    user: str
    password: str


class AlarmConfig(BaseModel):
    channel: str = "email"
    parameters: EmailParameters
    recipients: List[str]


class BaseOpenConfig(BaseModel):
    resolution: int = 3
    no_entry_hours: List[int] = []
    type: ConfigType = ConfigType.open


class BaseCloseConfig(BaseModel):
    type: ConfigType = ConfigType.close


class BaseTradeConfig(BaseModel):
    broker: MT5Broker
    data_source: DataSourceBroker
    action_methods: List[ActionMethod] = [ActionMethod.trade]
    alarm_config: Union[AlarmConfig, None] = None
    broker_stop_coefficient: float
    broker_take_profit: Union[float, None] = None
    deviation: int = 50
    open_config: Any = None
    close_config: Any = None
    time_to_trade: Union[time, None] = None


class BasePairConfig(BaseTradeConfig, Symbol):
    symbol: str


class MakeTradingStepResponse(BaseModel):
    time_to_sleep: int
    is_success: bool


class CheckedConfigResponse(BaseModel):
    applied_config: Any
    available_actions: List[int]


class BaseActionDetails(BaseModel):
    curr_time: datetime.datetime
    deal_size: Union[float, None] = None


class BuySellActionDetails(BaseActionDetails):
    price: float = None
    positive_only: bool = False


class ModifySLActionDetails(BaseActionDetails):
    new_sls: List[float]
    identifiers: List[int]


ActionDetails = Union[BaseActionDetails, BuySellActionDetails, ModifySLActionDetails]
