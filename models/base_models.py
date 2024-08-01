from pydantic import BaseModel, FilePath
from pair.enums import DataSource, Direction, ConfigType
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


class BaseOpenConfig(BaseModel):
    resolution: int = 3
    type: ConfigType = ConfigType.open


class BaseCloseConfig(BaseModel):
    type: ConfigType = ConfigType.close


class BaseTradeConfig(BaseModel):
    broker: MT5Broker
    data_source: DataSourceBroker
    broker_stop_coefficient: float
    broker_take_profit: float = None
    deviation: int = 50
    open_config: Any = None
    close_config: Any = None


class BasePairConfig(BaseTradeConfig, Symbol):
    symbol: str


class MakeTradingStepResponse(BaseModel):
    time_to_sleep: int
    is_success: bool


class CheckedConfigResponse(BaseModel):
    applied_config: Any
    available_actions: List[int]
