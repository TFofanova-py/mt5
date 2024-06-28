from pydantic import BaseModel, FilePath
from pair.enums import DataSource, Direction, DivegenceType
from typing import Union, Any


class Symbol(BaseModel):
    ds_symbol: str
    deal_size: float = 1.0
    direction: Direction = Direction.low_long


class MinNumberOfDibergence(BaseModel):
    entry: int = 1
    exit_tp: int = 1


class RsiIndicator(BaseModel):
    rsi_length: int


class MacdIndicator(BaseModel):
    fast_length: int
    slow_length: int
    signal_length: int


class BaseIndicator(BaseModel):
    length: int


class StkIndicator(BaseModel):
    stoch_length: int
    sma_length: int


class VwmacdIndicator(BaseModel):
    fast_length: int
    slow_length: int


Indicator = Union[
    RsiIndicator,
    MacdIndicator,
    BaseIndicator,
    StkIndicator,
    VwmacdIndicator,
]


class OpenConfig(BaseModel):
    resolution: int = 3
    pivot_period: int = 5
    entry: dict  # dict[str, Indicator]
    bollinger: Union[Any, None] = None
    next_position_bol_check: bool = False


class CloseConfig(BaseModel):
    resolution: int = 3
    pivot_period: int = 5
    positive_only: bool = False
    bot_stop_coefficient: Union[float, None] = None
    entry: dict  # dict[str, Indicator]


class BaseConfig(BaseModel):
    login: int
    password: str
    server: str
    path: FilePath
    data_source: DataSource = DataSource.mt5
    broker_stop_coefficient: float = 0.9995
    deviation: int = 50
    max_pivot_points: int = 10
    max_bars_to_check: int = 100
    dont_wait_for_confirmation: bool = True
    min_number_of_divergence: MinNumberOfDibergence
    divergence_type: DivegenceType = DivegenceType.regular
    open_config: OpenConfig
    close_config: CloseConfig
    entry_price_higher_than: Union[float, None] = None
    entry_price_lower_than: Union[float, None] = None
    exit_target: Union[float, None] = None


class BotConfig(BaseConfig):
    capital_creds: Union[dict, None] = None
    symbol_parameters: dict  # dict[str, Symbol]


class PairConfig(BaseConfig, Symbol):
    symbol: str
