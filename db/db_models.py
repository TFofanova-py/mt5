from sqlalchemy import Column, Integer, String, TIMESTAMP, Text, DECIMAL, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship


Base = declarative_base()

class IchimokuTrend(Base):
    __tablename__ = 'ichimoku_trend'

    id = Column(Integer, primary_key=True, index=True, unique=True)
    name = Column(String(20), nullable=False)

    trade = relationship("TrendTrade", back_populates="trend")

class Trade(Base):
    __tablename__ = 'trade'

    id = Column(Integer, primary_key=True, index=True, unique=True)
    opened_at = Column(TIMESTAMP, default="NOW()")
    broker_order=Column(String(30))
    symbol = Column(String(10), nullable=False)
    side = Column(String(10), nullable=False)
    open_price = Column(DECIMAL, nullable=False)
    volume = Column(DECIMAL, nullable=False)
    open_stop_loss = Column(DECIMAL)
    closed_at = Column(TIMESTAMP)
    close_price = Column(DECIMAL)
    close_reason_id = Column(Integer, ForeignKey('close_reason.id'))
    profit = Column(DECIMAL)

    reason = relationship("CloseReason", back_populates="trade")
    trend = relationship("TrendTrade", back_populates="trade")

class CloseReason(Base):
    __tablename__ = 'close_reason'

    id = Column(Integer, primary_key=True, index=True, unique=True)
    name = Column(String)

    trade = relationship("Trade", back_populates="reason")

class Divergence(Base):
    __tablename__ = 'divergence'

    id = Column(Integer, primary_key=True, index=True, unique=True)
    trade_id = Column(Integer, ForeignKey('trade.id'), nullable=False)
    indicator_id = Column(Integer, ForeignKey('indicator.id'), nullable=False)
    action_id = Column(Integer, ForeignKey('action.id'), nullable=False)
    side_id = Column(Integer, ForeignKey('divergence_side.id'), nullable=False)


class Action(Base):
    __tablename__ = 'action'

    id = Column(Integer, primary_key=True, index=True, unique=True)
    name = Column(String(10), nullable=False)

    # trade = relationship("IndicatorTrade", back_populates="action")


class DivergenceSide(Base):
    __tablename__ = 'divergence_side'

    id = Column(Integer, primary_key=True, index=True, unique=True)
    name = Column(String(10), nullable=False)

    # divergence = relationship("Divergence", back_populates="side")


class TrendTrade(Base):
    __tablename__ = 'trend_trade'

    id = Column(Integer, primary_key=True, index=True, unique=True)
    trade_id = Column(Integer, ForeignKey('trade.id'), nullable=False)
    trend_id = Column(Integer, ForeignKey('ichimoku_trend.id'), nullable=False)
    action_id = Column(Integer, ForeignKey('action.id'), nullable=True)
    trend_tf = Column(Integer, nullable=False)

    trade = relationship("Trade", back_populates="trend")
    trend = relationship("IchimokuTrend", back_populates="trade")

class IndicatorGroup(Base):
    __tablename__ = 'indicator_group'

    id = Column(Integer, primary_key=True, index=True, unique=True)
    name = Column(String(100), nullable=False)

    indicator = relationship("Indicator", back_populates="group")

class Indicator(Base):
    __tablename__ = 'indicator'

    id = Column(Integer, primary_key=True, index=True, unique=True)
    name = Column(String(30))
    group_id = Column(Integer, ForeignKey('indicator_group.id'), nullable=False)

    # One-to-many relationship with payments
    group = relationship("IndicatorGroup", back_populates="indicator")



