from enum import Enum


class DataSource(str, Enum):
    mt5 = "mt5"
    capital = "capital"
    yahoo = "yahoo"
    twelvedata = "twelvedata"


class DivegenceType(str, Enum):
    regular = "Regular"
    both = "Regular/Hidden"
    hidden = "Hidden"


class DivergenceMode(str, Enum):
    pos_reg = "positive_regular"
    neg_reg = "negative_regular"
    pos_hid = "positive_hidden"
    neg_hid = "negative_hidden"


class Direction(str, Enum):
    low_long = "low-long"
    high_short = "high-short"
    bi = "bi"
    swing = "swing"


class PriceDirection(str, Enum):
    up = "up"
    down = "down"


class RebuyCondition(str, Enum):
    short = "candle_close_higher_than_first_deal_price"
    long = "candle_close_lower_than_first_deal_price"


class ConfigType(str, Enum):
    open = "open"
    close = "close"


class ActionMethod(str, Enum):
    trade = "trade"
    alarm = "alarm"


class IchimokuTrend(str, Enum):
    strong_bullish = "Strong Bullish"
    bullish = "Bullish"
    strong_bearish = "Strong Bearish"
    bearish = "Bearish"
    consolidation = "Consolidation"