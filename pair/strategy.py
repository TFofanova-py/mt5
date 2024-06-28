import datetime
import logging
import pandas as pd
import numpy as np
from typing import List, Tuple
from .ta_utils import (
    rsi, macd, momentum, cci, obv,
    stk, vwmacd, cmf, mfi,
    pivotlow, pivothigh, bollinger_bands)
from .enums import DivegenceType, DivergenceMode, ConfigType, Direction
import MetaTrader5 as mt5
from models import PairConfig


class MultiIndStrategy:
    def __init__(self, p_config: PairConfig):
        self.pivot_period: int = p_config.open_config.pivot_period
        self.divtype = p_config.divergence_type
        self.min_number_of_divergence = p_config.min_number_of_divergence
        self.max_pivot_points = p_config.max_pivot_points
        self.max_bars_to_check = p_config.max_bars_to_check
        self.dont_wait_for_confirmation = p_config.dont_wait_for_confirmation
        self.indicators = p_config.open_config.entry
        self.direction = p_config.direction
        self.entry_price_higher_than = p_config.entry_price_higher_than
        self.entry_price_lower_than = p_config.entry_price_lower_than
        self.exit_target = p_config.exit_target
        self.last_divergence = {ConfigType.open: {"top": None, "bottom": None},
                                ConfigType.close: {"top": None, "bottom": None}}
        self.bollinger = p_config.open_config.bollinger
        self.close_positive_only = p_config.close_config.positive_only
        self.next_position_bol_check = p_config.open_config.next_position_bol_check
        self.bot_stop_coefficient = p_config.close_config.bot_stop_coefficient

    @staticmethod
    def arrived_divergence(src, close, startpoint, length, np_func):
        virtual_line_src = np.linspace(src.iloc[-length - 1], src.iloc[-startpoint - 1],
                                       length - startpoint)
        virtual_line_close = np.linspace(close.iloc[-length - 1], close.iloc[-startpoint - 1],
                                         length - startpoint)

        return all(np_func(src.iloc[-length - 1: -startpoint - 1], virtual_line_src)) and \
            all(np_func(close.iloc[-length - 1: -startpoint - 1], virtual_line_close))

    def divergence_length(self, src: pd.Series, close: pd.Series,
                          pivot_vals: np.array, pivot_positions: np.array,
                          mode: DivergenceMode):

        def is_suspected():
            func_src = np.greater if mode in [DivergenceMode.pos_reg, DivergenceMode.neg_hid] else np.less
            func_close = np.less if mode in [DivergenceMode.pos_reg, DivergenceMode.neg_hid] else np.greater

            return func_src(src.iloc[-startpoint - 1], src.iloc[-length - 1]) and \
                func_close(close.iloc[-startpoint - 1], pivot_vals.iloc[-x - 1])

        divlen = 0

        confirm_func = np.greater if mode in [DivergenceMode.pos_reg, DivergenceMode.pos_hid] else np.less
        scr_or_close_confirm = confirm_func(src.iloc[-1], src.iloc[-2]) and confirm_func(close.iloc[-1], close.iloc[-2])

        if self.dont_wait_for_confirmation or scr_or_close_confirm:
            startpoint = 0 if self.dont_wait_for_confirmation else 1

            for x in range(0, min(len(pivot_positions), self.max_pivot_points)):
                length = src.index[-1] - pivot_positions.iloc[-x - 1] + self.pivot_period

                # if we reach non valued array element or arrived 101. or previous bars then we don't search more
                if pivot_positions.iloc[-x - 1] == 0 or length > self.max_bars_to_check - 1:
                    break

                if length > 5 and is_suspected():
                    arrived_func = np.greater_equal if mode in [DivergenceMode.pos_reg, DivergenceMode.pos_hid] \
                        else np.less_equal

                    arrived = self.arrived_divergence(src, close, startpoint, length, arrived_func)

                    if arrived:
                        divlen = length
                        break

        return divlen

    def calculate_divs(self, indicator: pd.Series, close: pd.Series,
                       pl_vals: np.array, pl_positions: np.array,
                       ph_vals: np.array, ph_positions: np.array) -> np.array:
        divs = np.zeros(4, dtype=int)

        if self.divtype in [DivegenceType.regular, DivegenceType.both]:
            divs[0] = self.divergence_length(indicator, close, pl_vals, pl_positions, DivergenceMode.pos_reg)
            divs[1] = self.divergence_length(indicator, close, ph_vals, ph_positions, DivergenceMode.neg_reg)

        if self.divtype in [DivegenceType.hidden, DivegenceType.both]:
            divs[2] = self.divergence_length(indicator, close, pl_vals, pl_positions, DivergenceMode.pos_hid)
            divs[3] = self.divergence_length(indicator, close, ph_vals, ph_positions, DivergenceMode.neg_hid)

        return divs

    def count_divergence(self, data: pd.DataFrame, config_type: ConfigType) -> dict:

        ind_ser: List[pd.Series] = []
        indices: List[str] = []

        for ind_name, ind_params in self.indicators.items():
            ind_params = ind_params  # ind_params.dict()

            if ind_name == "rsi":
                ind_ser.append(rsi(data, period=ind_params["rsi_length"])["rsi"].reset_index(drop=True))
                indices.append("rsi")

            elif ind_name == "macd":
                macd_df = macd(data, **ind_params)
                ind_ser.append(macd_df["macd"].reset_index(drop=True))
                ind_ser.append(macd_df["hist"].reset_index(drop=True))
                indices.extend(["macd", "deltamacd"])

            elif ind_name == "momentum":
                ind_ser.append(momentum(data, **ind_params)["momentum"].reset_index(drop=True))
                indices.append("momentum")

            elif ind_name == "cci":
                ind_ser.append(cci(data, **ind_params)["cci"].reset_index(drop=True))
                indices.append("cci")

            elif ind_name == "obv":
                ind_ser.append(obv(data)["obv"].reset_index(drop=True))
                indices.append("obv")

            elif ind_name == "stk":
                ind_ser.append(stk(data, **ind_params)["stk"].reset_index(drop=True))
                indices.append("stk")

            elif ind_name == "vwmacd":
                ind_ser.append(vwmacd(data, **ind_params)["vwmacd"].reset_index(drop=True))
                indices.append("vwmacd")

            elif ind_name == "cmf":
                ind_ser.append(cmf(data, **ind_params)["cmf"].reset_index(drop=True))
                indices.append("cmf")

            elif ind_name == "mfi":
                ind_ser.append(mfi(data, **ind_params)["mfi"].reset_index(drop=True))
                indices.append("mfi")

        all_divergences = pd.DataFrame(np.zeros((len(ind_ser), 4)), index=indices)

        pl_vals, pl_positions = pivotlow(data["close"], self.pivot_period)
        ph_vals, ph_positions = pivothigh(data["close"], self.pivot_period)

        for i, curr_ind_ser in enumerate(ind_ser):
            all_divergences.iloc[i, :] = self.calculate_divs(curr_ind_ser[-self.max_bars_to_check:],
                                                             data["close"][-self.max_bars_to_check:],
                                                             pl_vals, pl_positions,
                                                             ph_vals, ph_positions)

        n_indicators = all_divergences.shape[0]

        div_types = (np.arange(4).reshape(1, -1) % 2) * np.ones((n_indicators, 1)).astype(int)

        top_mask = (div_types == 1)
        bottom_mask = (div_types == 0)

        div_signals = pd.DataFrame(np.zeros((n_indicators, 2)),
                                   index=indices, columns=["top", "bottom"])
        div_signals.iloc[:, 0] = np.any((all_divergences > 0) * top_mask, axis=1).astype(int)
        div_signals.iloc[:, 1] = np.any((all_divergences > 0) * bottom_mask, axis=1).astype(int)

        # update last_divergence and return
        for t in ["top", "bottom"]:
            if div_signals[t].sum() > 0:
                # if it was a pivot high/ low after last top/ bottom divergence
                new_divergence = True
                if self.last_divergence[config_type][t] is not None and self.last_divergence[config_type][t] >= data.index[0]:
                    idx = data.index.get_loc(self.last_divergence[config_type][t])

                    new_divergence = (pl_positions.iloc[-1] > idx) if t == "bottom" else (ph_positions.iloc[-1] > idx)

                if new_divergence:
                    self.last_divergence[config_type][t] = data.index[-1]
                    triggered_idx = np.any(div_signals > 0, axis=1).tolist()
                    triggered_inds = [idx_name for idx, idx_name in enumerate(indices) if triggered_idx[idx]]

                    return {"top": div_signals["top"].sum(),
                            "bottom": div_signals["bottom"].sum(), "triggered": triggered_inds}

        return {"top": 0, "bottom": 0, "triggered": []}

    @staticmethod
    def was_price_goes_up(data: pd.DataFrame) -> bool:
        assert "close" in data.columns
        assert data.shape[0] > 1
        if data["close"].iloc[-1] > data["close"].iloc[-2]:
            return True
        return False

    def get_bollinger_conditions(self, data: pd.DataFrame) -> Tuple[bool, bool]:
        price = data["close"].iloc[-1]
        if self.bollinger is None:
            return True, True
        params = self.bollinger
        ind_bollinger = bollinger_bands(data, **params).iloc[-1][["upper", "lower"]]
        return price < ind_bollinger["lower"], price > ind_bollinger["upper"]

    def get_action(self, data: pd.DataFrame,
                   symbol: str,
                   positions: list,
                   stop_coefficient: float,
                   trade_tick_size: float,
                   config_type: ConfigType) -> Tuple[int, dict]:
        divergences_cnt = self.count_divergence(data, config_type=config_type)

        type_action = None
        details = {"curr_time": data.index[-1]}
        price = data["close"].iloc[-1]

        bollinger_cond_lower, bollinger_cond_upper = self.get_bollinger_conditions(data)

        if divergences_cnt["top"] + divergences_cnt["bottom"] > 0:
            print(f"{datetime.datetime.now().time().isoformat(timespec='minutes')} {symbol}: divergences {divergences_cnt}, bollinger_lower: {bollinger_cond_lower}, bollinger_upper: {bollinger_cond_upper}")
            logging.info(f"{data.index[-1]}, divergences: {divergences_cnt}, bollinger_lower: {bollinger_cond_lower}, bollinger_upper: {bollinger_cond_upper}")

        if self.direction in [Direction.low_long, Direction.bi, Direction.swing] and \
                divergences_cnt["bottom"] >= self.min_number_of_divergence.entry and \
                bollinger_cond_lower and \
                len(positions) == 0 and \
                (self.direction != Direction.low_long or
                 self.entry_price_lower_than is None or
                 (self.entry_price_lower_than and price < self.entry_price_lower_than)):
            type_action = mt5.ORDER_TYPE_BUY

        elif self.direction in [Direction.high_short, Direction.bi, Direction.swing] and \
                divergences_cnt["top"] >= self.min_number_of_divergence.entry and \
                bollinger_cond_upper and \
                len(positions) == 0 and \
                (self.direction != Direction.high_short or
                 self.entry_price_higher_than is None or
                 (self.entry_price_higher_than and price > self.entry_price_higher_than)):
            type_action = mt5.ORDER_TYPE_SELL

        elif len(positions) > 0:
            same_direction_divergences = divergences_cnt["bottom"] if positions[0].type == 0 \
                else divergences_cnt["top"]
            opposite_direction_divergences = divergences_cnt["top"] if positions[0].type == 0 else divergences_cnt[
                "bottom"]

            print(datetime.datetime.now().time().isoformat(timespec='minutes'), symbol, self.direction, "len pos", len(positions),
                  "same direction", same_direction_divergences, "opposite direction", opposite_direction_divergences)

            func_stop = np.less if positions[0].type == 0 else np.greater
            if func_stop(positions[0].price_current / positions[0].price_open, self.bot_stop_coefficient):
                print(f'{symbol}: Close position because of bot stop {self.bot_stop_coefficient}')
                type_action = (positions[0].type + 1) % 2

            elif opposite_direction_divergences >= self.min_number_of_divergence.exit_tp and (
                    self.exit_target is None or (self.direction == Direction.high_short) * (price < self.exit_target)
                    or (self.direction == Direction.low_long) * (price > self.exit_target)):
                # close position
                type_action = (positions[0].type + 1) % 2
                details.update({"positive_only": self.close_positive_only})

            elif same_direction_divergences >= self.min_number_of_divergence.entry:
                bollinger_cond = None
                if self.next_position_bol_check:
                    bollinger_cond = (positions[0].type == 0 and bollinger_cond_lower) or (
                            positions[0].type == 1 and bollinger_cond_upper)
                    print(
                        f"{datetime.datetime.now().time().isoformat(timespec='minutes')} {symbol}: bollinger common: {bollinger_cond}")

                if not self.next_position_bol_check or bollinger_cond:
                    # open another position
                    type_action = positions[0].type
                    print(
                        f"{datetime.datetime.now().time().isoformat(timespec='minutes')} {symbol}: type_action: {type_action}")
            else:
                # modify stop-loss
                price_goes_up = self.was_price_goes_up(data)
                new_sls = None

                if positions[0].type == 0 and price_goes_up:
                    # if long and price goes up, move sl up
                    new_sl = round(price * stop_coefficient, abs(int(np.log10(trade_tick_size))))
                    new_sls = [new_sl if x.sl < new_sl else None for x in positions]
                elif positions[0].type == 1 and not price_goes_up:
                    # if short and price goes down, move sl down
                    new_sl = round(price * (2. - stop_coefficient), abs(int(np.log10(trade_tick_size))))
                    new_sls = [new_sl if x.sl > new_sl else None for x in positions]

                if new_sls is not None:
                    type_action = mt5.TRADE_ACTION_SLTP
                    details.update({"new_sls": new_sls})

        if type_action in [mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_SELL]:
            details.update({"price": data["close"].iloc[-1]})

        return type_action, details
