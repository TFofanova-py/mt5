from pydantic import BaseModel
from .base_models import MT5Broker, DataSourceBroker, BaseOpenConfig, BaseCloseConfig
from pair.enums import Direction, PriceDirection, RebuyCondition
from typing import Union, List


class RebuyConfig(BaseModel):
    is_allowed: bool
    deal_size: float = 1.0
    condition: RebuyCondition


class RelatedOpenConfig(BaseOpenConfig):
    num_candles_in_row: int
    candle_minutes: int
    candle_direction: PriceDirection
    trigger_for_deal: float
    rebuy_config: RebuyConfig


class RelatedCloseConfig(BaseCloseConfig):
    bot_take_profit: Union[float, None] = None


class TradeConfig(BaseModel):
    ticker_to_monitor: str
    ticker_to_trade: str
    deal_size: float = 1.0
    deal_direction: Direction = Direction.low_long
    broker_stop_coefficient: float = 0.98
    broker_take_profit: float = 1.05
    deviation: int = 50
    open_config: RelatedOpenConfig
    close_config: RelatedCloseConfig


class BotConfig(BaseModel):
    broker: MT5Broker
    data_source: DataSourceBroker
    trade_configs: List[TradeConfig]


class RelatedPairConfig(TradeConfig):
    broker: MT5Broker
    data_source: DataSourceBroker
