from pydantic import BaseModel
from .base_models import BaseTradeConfig, BaseOpenConfig, BaseCloseConfig, Symbol, BuySellActionDetails
from pair.enums import DivegenceType, IchimokuTrend, IchimokuLayerStatus, ClosePositionReason
from typing import Union, Any, Tuple, List


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


class IchimokuLayer(BaseModel):
    tf: int
    status: IchimokuLayerStatus


class IchimokuConfig(BaseModel):
    layers: List[IchimokuLayer]
    periods: Tuple[int, int, int]

class IchimokuResponse(BaseModel):
    trends: List[Union[IchimokuTrend, None]]
    is_changing: bool

class OpenConfig(BaseOpenConfig):
    pivot_period: int = 5
    ichimoku: Union[IchimokuConfig, None] = None
    entry: dict
    bollinger: Union[Any, None] = None
    next_position_bol_check: bool = False
    max_position_count: Union[int, None] = None


class CloseConfig(BaseCloseConfig):
    resolution: int = 3
    pivot_period: int = 5
    positive_only: bool = False
    bot_stop_coefficient: Union[float, None] = None
    entry: dict


class BaseConfig(BaseTradeConfig):
    max_pivot_points: int = 10
    max_bars_to_check: int = 100
    dont_wait_for_confirmation: bool = True
    min_number_of_divergence: MinNumberOfDibergence
    divergence_type: DivegenceType = DivegenceType.regular
    entry_price_higher_than: Union[float, None] = None
    entry_price_lower_than: Union[float, None] = None
    exit_target: Union[float, None] = None
    open_config: OpenConfig = None
    close_config: CloseConfig = None


class BotConfig(BaseConfig):
    capital_creds: Union[dict, None] = None
    symbol_parameters: dict  # dict[str, Symbol]


class PairConfig(BaseConfig, Symbol):
    symbol: str

class DivergenceCountResponse(BaseModel):
    top_cnt: int = 0
    bottom_cnt: int = 0
    top_triggered: List[str] = []
    bottom_triggered: List[str] = []

class IchimokuTrendResponse(BaseModel):
    long: IchimokuTrend
    short: IchimokuTrend
    long_tf: int
    short_tf: int

class MultiIndBuySellActionDetails(BuySellActionDetails):
    divergence: DivergenceCountResponse = None
    ichimoku_trends: IchimokuTrendResponse = None
    reason: ClosePositionReason = None



