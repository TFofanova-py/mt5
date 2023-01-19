import pandas as pd
import numpy as np
import pytest
import MetaTrader5 as mt5
from config_mt5 import Config
from pair import Pair
from main import create_position, close_opened_position
from time import sleep


@pytest.fixture(scope="module")
def session():
    mt5.initialize(login=Config.login, password=Config.password, server=Config.server, path=Config.path)
    yield {"broker": mt5,
           "pairs": {"first_pair_params": {"symbol": "[SP500]",
                                           "resolution": mt5.TIMEFRAME_M1,
                                           "deal_size": 1.0,
                                           "stop_coefficient": 0.997,
                                           "limit": None,
                                           "dft_period": 19},
                     "second_pair_params": {"symbol": "USDCHF",
                                            "resolution": mt5.TIMEFRAME_M3,
                                            "deal_size": 0.5,
                                            "stop_coefficient": 0.998,
                                            "limit": None,
                                            "dft_period": 34},
                     "pair_with_wrong_deal_size_params": {"symbol": "[SP500]",
                                                          "resolution": mt5.TIMEFRAME_M1,
                                                          "deal_size": 1000.0,
                                                          "stop_coefficient": 0.994,
                                                          "limit": None,
                                                          "dft_period": 19},
                     "pair_with_wrong_stop_params": {"symbol": "BTCUSD",
                                                     "resolution": mt5.TIMEFRAME_M3,
                                                     "deal_size": 0.5,
                                                     "stop_coefficient": 0.999,
                                                     "limit": None,
                                                     "dft_period": 34}
                     }
           }


@pytest.fixture(scope="function")
def make_pairs(session):
    pairs_dict = {}
    for pair_key in session["pairs"]:
        pair_params = session["pairs"][pair_key]
        pair_key_short = pair_key[:-7]
        pairs_dict[pair_key_short] = Pair(symbol=pair_params["symbol"],
                                          resolution=pair_params["resolution"],
                                          deal_size=pair_params["deal_size"],
                                          stop_coef=pair_params["stop_coefficient"],
                                          limit_coef=pair_params["limit"],
                                          dft_period=pair_params["dft_period"])

    yield pairs_dict


@pytest.fixture(scope="function")
def buy_two_instruments(session, make_pairs):
    broker = session["broker"]
    first_pair = make_pairs["first_pair"]
    first_pair.fetch_prices(broker, numpoints=10)
    create_position(broker, first_pair)

    second_pair = make_pairs["second_pair"]
    second_pair.fetch_prices(broker, numpoints=10)
    create_position(broker, second_pair)

    open_pos = broker.positions_total()
    assert open_pos >= np.floor(first_pair.deal_size) + np.floor(second_pair.deal_size)
    yield {"first_pair": first_pair,
           "second_pair": second_pair}


def test_check_order(session, make_pairs):
    broker = session["broker"]

    wrong_pairs = (make_pairs["pair_with_wrong_deal_size"],
                   make_pairs["pair_with_wrong_stop"])

    assert make_pairs["pair_with_wrong_deal_size"] is not None
    assert make_pairs["pair_with_wrong_stop"] is not None

    for pair in wrong_pairs:
        pair.fetch_prices(broker, numpoints=10)

        response = create_position(broker, pair)
        assert response is None


def test_create_position(session, make_pairs):
    broker = session["broker"]
    pair = make_pairs["first_pair"]
    pair.fetch_prices(broker, numpoints=10)

    response = create_position(broker, pair)
    assert response is not None

    try:
        assert response.order != 0
        assert response.volume == pair.deal_size
    except Exception as e:
        print(e)
    sleep(20)


def test_close_opened_position(session, make_pairs):
    broker = session["broker"]
    pair = make_pairs["first_pair"]
    open_pos = broker.positions_total()
    assert open_pos > 0
    if pair.prices is None:
        pair.fetch_prices(broker, numpoints=10)

    response = close_opened_position(broker, pair)
    assert response is not None
    assert response.volume == pair.deal_size


def test_close_first_buyed_position(session, buy_two_instruments):
    broker = session["broker"]
    first_pair = buy_two_instruments["first_pair"]
    second_pair = buy_two_instruments["second_pair"]

    sleep(20)

    response = close_opened_position(broker, first_pair)
    assert response is not None
    assert np.allclose(response.volume, first_pair.deal_size)

    response = close_opened_position(broker, second_pair)
    assert response is not None
    assert np.allclose(response.volume, second_pair.deal_size)


def test_close_second_buyed_position(session, buy_two_instruments):
    broker = session["broker"]
    first_pair = buy_two_instruments["first_pair"]
    second_pair = buy_two_instruments["second_pair"]

    sleep(20)

    response = close_opened_position(broker, second_pair)
    assert response is not None
    assert np.allclose(response.volume, second_pair.deal_size)

    response = close_opened_position(broker, first_pair)
    assert response is not None
    assert np.allclose(response.volume, first_pair.deal_size)


def test_fetch_prices(session, make_pairs):
    broker = session["broker"]
    pair = make_pairs["first_pair"]

    # initial fetching
    pair.fetch_prices(broker)
    pair.prices.iloc[-1, 0] = np.nan

    sleep(60)

    # adding prices
    pair.fetch_prices(broker, numpoints=10)

    assert pair.prices.isna().sum().sum() == 0

# #  python -m pytest test_open_close.py
# # --trace - option for debug
# #  s | n - next, s - step into

