import datetime

import pandas as pd
import pytest
from Simulaton.simulate import was_stoploss, simulation_step, update_pair, update_balance, run_simulate
from pair import Pair
from unittest import mock
import numpy as np


@pytest.fixture(scope="function")
def make_simulated_pair():
    pair = Pair(symbol="ES=F", resolution=3, deal_size=1.0,
                stop_coef=0.995, limit_coef=None, dft_period=34,
                highest_fib=0.220, lowest_fib=0.800, n_down_periods=[3, 4, np.inf])

    pair.prices = get_test_data(pair.symbol)

    yield {"simulated_pair": pair}


def get_test_data(symbol: str) -> pd.DataFrame:
    data = pd.read_csv("../Tests/SP500_yahoo_data.csv", index_col=0)
    data.index = pd.to_datetime(data.index).tz_localize(None)
    return data


def test_was_stoploss(make_simulated_pair):
    pair = make_simulated_pair["simulated_pair"]
    pair.prices.iloc[-1, -1] = 3500.0
    assert was_stoploss(pair, {"cash": -10.9, "position": {"level": 3999.5, "stop_level": 3550.4, "size": 1}})
    assert not was_stoploss(pair, {"cash": -10.9, "position": {"level": 3999.5, "stop_level": 3000.4, "size": 1}})
    assert not was_stoploss(pair, {"cash": -10.9, "position": {"level": 3999.5, "stop_level": 3600.7, "size": 0}})
    assert not was_stoploss(pair, {"cash": -10.9, "position": {"level": 3999.5, "stop_level": 3000.7, "size": 0}})


def test_update_balance(make_simulated_pair):
    pair = make_simulated_pair["simulated_pair"]
    pair.prices.iloc[-1, -1] = 3500.0

    # buy
    balance = {"cash": 100.0, "position": {"level": None, "stop_level": None, "size": 0}}
    updated_balance = update_balance(pair, balance, 1)
    assert updated_balance["cash"] + 3400.0 < 1e-5
    assert updated_balance["position"]["size"] == 1
    assert updated_balance["position"]["level"] - 3500.0 < 1e-5
    assert updated_balance["position"]["stop_level"] - pair.stop_coefficient * 3500.0 < 1e-5

    # sell
    balance = {"cash": -3400.0, "position": {"level": 3450.5, "stop_level": 3400.0, "size": 1}}
    updated_balance = update_balance(pair, balance, -1)
    assert updated_balance["cash"] - 100.0 < 1e-5
    assert updated_balance["position"]["size"] == 0
    assert updated_balance["position"]["level"] is None
    assert updated_balance["position"]["stop_level"] is None

    # hold
    balance = {"cash": -3400.0, "position": {"level": 3450.5, "stop_level": 3400.0, "size": 1}}
    updated_balance = update_balance(pair, balance, 0)
    assert updated_balance["cash"] - balance["cash"] < 1e-5
    assert updated_balance["position"]["size"] == balance["position"]["size"]
    assert updated_balance["position"]["level"] - 3500.0 < 1e-5
    assert updated_balance["position"]["stop_level"] == balance["position"]["stop_level"]

    # stop loss
    balance = {"cash": -3400.0, "position": {"level": 3650.5, "stop_level": 3550.0, "size": 1}}
    updated_balance = update_balance(pair, balance, -2)
    assert updated_balance["cash"] - 100.0 < 1e-5
    assert updated_balance["position"]["size"] == 0
    assert updated_balance["position"]["level"] is None
    assert updated_balance["position"]["stop_level"] is None


def test_update_pair(make_simulated_pair):
    pair = make_simulated_pair["simulated_pair"]
    pair.prices.iloc[-1, -1] = 3500.0
    t_last = pair.prices.index[-1]

    # buy
    balance = {"cash": 100.0, "position": {"level": None, "stop_level": None, "size": 0}}
    updated_pair = update_pair(pair, balance, 1)
    assert updated_pair == pair

    # sell
    balance = {"cash": -3400.0, "position": {"level": 3450.5, "stop_level": 3400.0, "size": 1}}
    updated_pair = update_pair(pair, balance, -1)
    assert updated_pair.last_sell == t_last
    assert updated_pair.idx_down_periods == 0

    # hold
    balance = {"cash": -3400.0, "position": {"level": 3450.5, "stop_level": 3400.0, "size": 1}}
    updated_pair = update_pair(pair, balance, 0)
    assert updated_pair == pair

    # stop loss
    balance = {"cash": -3400.0, "position": {"level": 3650.5, "stop_level": 3550.0, "size": 1}}
    updated_pair = update_pair(pair, balance, -2)
    assert updated_pair.last_sell == t_last
    assert updated_pair.idx_down_periods - pair.idx_down_periods <= 1


def test_simulation_step(make_simulated_pair):
    pair = make_simulated_pair["simulated_pair"]

    buy_date = datetime.datetime(year=2022, month=12, day=26, hour=22, minute=18)  # level 3894.5
    buy_prices = pair.prices[:buy_date]

    sell_date = datetime.datetime(year=2022, month=12, day=27, hour=5, minute=25)  # level 3896.0
    sell_prices = pair.prices[:sell_date]

    # buy
    pair.prices = buy_prices
    assert simulation_step(pair, {"cash": 100.0, "position": {"level": None, "stop_level": None, "size": 0}})[1] == 1

    assert simulation_step(pair,
                           {"cash": 100.0, "position": {"level": 4500.0, "stop_level": 4000.0, "size": 1}})[1] == -2

    assert simulation_step(pair,
                           {"cash": 100.0, "position": {"level": 3500.0, "stop_level": 3400.0, "size": 1}})[1] == 1

    # sell
    pair.prices = sell_prices
    assert simulation_step(pair,
                           {"cash": 100.0, "position": {"level": 3897.5, "stop_level": 3872.0, "size": 1}})[1] == -1

    assert simulation_step(pair, {"cash": 100.0, "position": {"level": None, "stop_level": None, "size": 0}})[1] == -1

    assert simulation_step(pair,
                           {"cash": 100.0, "position": {"level": 4500.0, "stop_level": 3900.0, "size": 1}})[1] == -2


@mock.patch("yahoo_utils.get_yahoo_data", side_effect=get_test_data)
def test_run_simulation(mock_dump_fn):
    assert run_simulate(symbol="ES=F", resolution=3, deal_size=1.0, stop_coefficient=0.995,
                        limit=None, dft_period=34, highest_fib=0.220, lowest_fib=0.800,
                        down_periods="[3, 4]") + 0.3 < 1e-5
