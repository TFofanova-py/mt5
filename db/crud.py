import datetime
from typing import Literal
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import create_engine, func, select
from db.db_models import Base, Indicator, Trade, Divergence, IchimokuTrend, \
    CloseReason, TrendTrade
from contextlib import contextmanager
from models.multiind_models import IchimokuTrendResponse, DivergenceCountResponse
from pair.enums import ClosePositionReason


class Database:
    def __init__(self):
        DATABASE_URL = "postgresql://postgres:a1s2d3f4@localhost:5432/algotrading"
        self.engine = create_engine(DATABASE_URL)
        self.session_maker = sessionmaker(self.engine, autoflush=False, autocommit=False)
        self.base = declarative_base()
        self.session = self.get_session()
        Base.metadata.create_all(self.engine)

    @contextmanager
    def get_session(self):
        with self.session_maker() as session:
            try:
                yield session
            finally:
               session.close()

    def insert_trade(self, symbol: str,
                     broker_order: str,
                     side: Literal["buy", "sell"],
                     open_price: float,
                     volume: float,
                     open_stop_loss: float,
                     divergence: DivergenceCountResponse = None,
                     ichimoku: IchimokuTrendResponse = None
                     ):
        trade_data = {
            "symbol": symbol,
            "broker_order": broker_order,
            "side": side,
            "open_price": open_price,
            "volume": volume,
            "open_stop_loss": open_stop_loss
        }

        with self.get_session() as session:
            new_trade = Trade(**trade_data)
            session.add(new_trade)
            session.flush()

            action_id = 1 # open position
            self.insert_divergence_and_ichimoku(session=session, trade_id=new_trade.id, action_id=action_id, divergence=divergence, ichimoku=ichimoku)
            session.commit()

    def close_trade(self, broker_order: str, reason: ClosePositionReason, close_price: float, profit: float,
                    divergence: DivergenceCountResponse = None,
                    ichimoku: IchimokuTrendResponse = None):
        with self.get_session() as session:
            data = {
                "closed_at": datetime.datetime.now(),
                "close_price": close_price,
                "close_reason_id": session.query(CloseReason.id).filter(CloseReason.name == reason.value).first()[0],
                "profit": profit
            }
            session.execute(
                Trade.__table__.update().where(Trade.broker_order == broker_order).values(**data)
            )
            session.flush()
            trade_id = session.query(Trade.id).filter(Trade.broker_order == broker_order).first()[0]

            action_id = 2  # close position
            self.insert_divergence_and_ichimoku(session=session, trade_id=trade_id, action_id=action_id, divergence=divergence,
                                                ichimoku=ichimoku)
            session.commit()

    def insert_divergence_and_ichimoku(self, session, trade_id: int, action_id: int, divergence: DivergenceCountResponse = None, ichimoku: IchimokuTrendResponse = None):
        if divergence is not None:
            indicators = divergence.top_triggered or divergence.bottom_triggered
            indicators = [x.lower() for x in indicators]
            side_id = 1 if divergence.bottom_cnt > 0 else 2
            indicator_ids = [x[0] for x in session.query(Indicator.id).filter(Indicator.name.in_(indicators)).all()]
            for ind in indicator_ids:
                new_divegence = Divergence(trade_id=trade_id, indicator_id=ind, action_id=action_id,
                                           side_id=side_id)
                session.add(new_divegence)
                session.flush()

        if ichimoku is not None:
            long_trend_id = session.query(IchimokuTrend.id).filter(IchimokuTrend.name == ichimoku.long.value).first()[0]
            new_trend = TrendTrade(trade_id=trade_id, trend_id=long_trend_id, trend_tf=ichimoku.long_tf, action_id=action_id)
            session.add(new_trend)

            short_trend_id = session.query(IchimokuTrend.id).filter(IchimokuTrend.name == ichimoku.short.value).first()[
                0]
            new_trend = TrendTrade(trade_id=trade_id, trend_id=short_trend_id, trend_tf=ichimoku.short_tf, action_id=action_id)
            session.add(new_trend)

    def insert_row(self):
        data = {
            "name": "Divergence"
        }
        with self.get_session() as session:
            new_row = CloseReason(**data)
            session.add(new_row)
            session.commit()
            print(new_row.id)

    def update_row(self):
        old_name = "MFI"
        data = {
            "action_id": 1
        }
        with self.get_session() as session:
            session.execute(
                TrendTrade.__table__.update().values(**data)
            )
            session.commit()
            rows = session.query(TrendTrade.id).all()
            print([x for x in rows])

    def select(self):
        with self.get_session() as session:
            rows = session.query(Divergence).all()
            print([(x.trade_id, x.side_id, x.action_id) for x in rows])

    def report(self):
        with (self.get_session() as session):
            divergences = (select(Divergence.trade_id,
                                  func.array_agg(Indicator.name).filter(Divergence.action_id == 1).label(
                                      "open_divergences"),
                                  func.array_agg(Indicator.name).filter(Divergence.action_id == 2).label(
                                      "close_divergences")
                                  )
                    .join(Indicator, Divergence.indicator_id == Indicator.id)
                    .group_by(Divergence.trade_id)
                    ).subquery()

            trends = (select(TrendTrade.trade_id,
                                  func.array_agg(func.json_build_array(TrendTrade.trend_tf, IchimokuTrend.name)).filter(TrendTrade.action_id == 1).label("open_trends"),
                                  func.array_agg(
                                      func.json_build_array(TrendTrade.trend_tf, IchimokuTrend.name)).filter(
                                      TrendTrade.action_id == 2).label("close_trends"),

                                  )
                    .join(IchimokuTrend, TrendTrade.trend_id == IchimokuTrend.id)
                    .group_by(TrendTrade.trade_id)
                    ).subquery()

            full_report = (select(Trade.id, Trade.broker_order, Trade.opened_at, Trade.closed_at, divergences.c.open_divergences, divergences.c.close_divergences, trends.c.open_trends, trends.c.close_trends)
            .join(divergences, divergences.c.trade_id == Trade.id)
            .join(trends, trends.c.trade_id == Trade.id)
                           .order_by(Trade.id))

            return session.execute(full_report).fetchall()

db_client = Database()

if __name__ == "__main__":
    # db_client.update_row()
    # db_client.insert_trade(symbol="test", broker_order="test_order", side="buy", open_price=1.0, volume=105.4, open_stop_loss=0.9,
    #                        divergence=DivergenceCountResponse(bottom_cnt=1, bottom_triggered=["RSI"]),
    #                        ichimoku=IchimokuTrendResponse(long=pair.enums.IchimokuTrend.consolidation, short=pair.enums.IchimokuTrend.bullish, long_tf=240, short_tf=60))
    # db_client.close_trade(broker_order="test_order", reason=ClosePositionReason.divergence, close_price=1.2, profit=0.2, divergence=DivergenceCountResponse(top_cnt=1, top_triggered=["Momentum"]),
    #                       ichimoku=IchimokuTrendResponse(long_tf=280, short_tf=120, long=pair.enums.IchimokuTrend.bullish, short=pair.enums.IchimokuTrend.strong_bullish))
    print(db_client.report())