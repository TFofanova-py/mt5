import logging
from pair.multiindpair import MultiIndPair
from models.multiind_models import PairConfig
from models.base_models import ConfigType
import numpy as np
from typing import Union, List
from pydantic import BaseModel
from pair.enums import Direction


class Position(BaseModel):
    type: int
    price_current: float = None
    price_open: float
    volume: float
    sl: float
    identifier: int


class TradeRequest(BaseModel):
    position: int


class OpenCloseResponse(BaseModel):
    order: int
    msg: str
    request: TradeRequest


class SimulatePair(MultiIndPair):
    def __init__(self, broker, pair_config: PairConfig):
        super().__init__(broker, pair_config)
        self.identifier: int = 0
        self.balance: float = None
        self.history = []

    def make_simulation_step(self, curr_time, open_data, close_data, balance) -> float:
        self.balance = balance

        # hundle stoplosses
        long_stop_losses = [x for x in self.positions if x.sl >= close_data.iloc[-1]["low"] and x.type == 0]
        short_stop_losses = [x for x in self.positions if x.sl <= close_data.iloc[-1]["high"] and x.type == 1]
        self.balance += (sum([x.sl * x.volume for x in long_stop_losses]) -
                         sum([x.sl * x.volume for x in short_stop_losses]))
        if long_stop_losses or short_stop_losses:
            msg = f"{self.symbol}: Stop losses: {long_stop_losses + short_stop_losses}"
            logging.info(msg)
            print(msg)

        # update positions
        self.positions = [x for x in self.positions if x not in long_stop_losses + short_stop_losses]
        self.orders = [x.identifier for x in self.positions]

        configs_to_check = self.get_configs_to_check(curr_time=curr_time)
        self.update_last_check_time(configs_to_check, curr_time=curr_time)

        for cnf in configs_to_check:
            data = open_data if cnf.applied_config.type == ConfigType.open else close_data
            for pos in self.positions:
                pos.price_current = data.iloc[-1]["close"]
            type_action, action_details = self.strategy.get_action(data=data,
                                                                   symbol=self.symbol,
                                                                   positions=self.positions,
                                                                   stop_coefficient=self.broker_stop_coefficient,
                                                                   trade_tick_size=self.trade_tick_size,
                                                                   config=cnf.applied_config,
                                                                   verbose=True)

            if type_action in cnf.available_actions:
                self.make_action(type_action, action_details, curr_time=curr_time)

            elif type_action is not None:
                logging.info(
                    f"{curr_time} {self.symbol}: Action {type_action} is not available for {cnf.applied_config.type.value} config and {len(self.positions)} positions and {self.strategy.direction} direction")

        return self.balance

    def create_position(self, price: float, type_action: int, sl: float = None, deal_size: float = None, **kwargs):
        volume = deal_size if deal_size else self.deal_size

        if sl is None:
            if type_action == 0:
                sl = round(price * self.broker_stop_coefficient, abs(int(np.log10(self.trade_tick_size))))
            else:
                sl = round(price * (2. - self.broker_stop_coefficient), abs(int(np.log10(self.trade_tick_size))))

        if self.balance < price * volume:
            print(
                f"Balance isn't enough. Balance: {self.balance}, symbol {self.symbol}, {price} * {self.deal_size}, {type_action}")
            return

        if type_action == 0:
            self.balance -= price * volume  # open long
        else:
            self.balance += price * volume  # open short

        self.identifier += 1
        self.positions.append(Position(type=type_action, price_open=price, volume=volume, sl=sl,
                                       identifier=self.identifier))
        self.orders.append(self.identifier)
        self.history.append((kwargs.get("curr_time"), self.symbol, price, type_action))
        return

    def close_opened_position(self, price: float, type_action: Union[str, int], identifiers: List[int] = None,
                              positive_only: bool = False, **kwargs) -> list:
        if identifiers is None:
            identifiers = self.orders

        profits = [(price - pos.price_open) * (-1)**(int(self.strategy.direction == Direction.high_short))
                   for pos in self.positions if pos.identifier in identifiers]

        responses = []
        for i, position in enumerate(identifiers):
            if not positive_only or profits[i] > 0:
                item = [x for x in self.positions if x.identifier == position][0]
                self.positions.remove(item)
                responses.append(OpenCloseResponse(order=position, msg="OK", request=TradeRequest(position=position)))
                self.history.append((kwargs.get("curr_time"), self.symbol, price, type_action))

                if type_action == 0:
                    self.balance -= price * item.volume  # close short
                else:
                    self.balance += price * item.volume  # close long

            else:
                responses.append(OpenCloseResponse(order=position, msg=f"Don't close negative position",
                                                   request=TradeRequest(position=position)))
        return responses

    def modify_sl(self, new_sls: List[float], identifiers: List[int] = None, **kwargs) -> list:
        if identifiers is None:
            identifiers = self.orders

        assert new_sls is not None
        assert len(identifiers) == len(new_sls)

        positions = [x for x in self.positions if x.identifier in identifiers]

        for position, new_sl in zip(positions, new_sls):
            position.sl = new_sl

        self.history.append((kwargs.get("curr_time"), self.symbol, positions[0].price_current, kwargs.get("type_action")))

        return [f"{self.symbol}, {i}, OK" for i in identifiers]


