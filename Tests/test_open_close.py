import json
import numpy as np
import pytest
import MetaTrader5 as mt5
from pair.multiindpair import BasePair
from time import sleep
from models.multiind_models import BotConfig, PairConfig


@pytest.fixture(scope="module")
def session():
    config = json.load(open("../test-ichimoku.json"))
    config = BotConfig(**config)
    initialization_config = {"login": config.broker.login,
                             "password": config.broker.password,
                             "server": config.broker.server,
                             "path": str(config.broker.path)}
    mt5.initialize(**initialization_config)
    yield {"broker": mt5,
           "pairs": {"first_pair_params": PairConfig(broker=initialization_config,
                                                     data_source={"name": "mt5", "connection": initialization_config},
                                                     **{"symbol": "USDCAD",
                                                         "ds_symbol": "USDCAD",
                                                        "deal_size": 0.05,
                                                        "broker_stop_coefficient": 0.997,
                                                        "direction": "low-long",
                                                        "max_pivot_points": 10,
                                                        "max_bars_to_check": 100,
                                                        "dont_wait_for_confirmation": False,
                                                        "min_number_of_divergence": {
                                                            "entry": 1,
                                                            "exit_sl": 1,
                                                            "exit_tp": 1
                                                        },
                                                        "divergence_type": "Regular",
                                                        "open_config": {
                                                            "resolution": mt5.TIMEFRAME_M1,
                                                            "pivot_period": 10,
                                                            "entry": {
                                                                "rsi": {
                                                                    "rsi_length": 14
                                                                },
                                                            }
                                                        },
                                                        "close_config": {
                                                            "resolution": mt5.TIMEFRAME_M1,
                                                            "pivot_period": 10,
                                                            "entry": {
                                                                "rsi": {
                                                                    "rsi_length": 14
                                                                },
                                                            }
                                                        }}),
                     "second_pair_params": PairConfig(broker=initialization_config,
                                                     data_source={"name": "mt5", "connection": initialization_config},
                                                      **{"symbol": "USDCHF",
                                                         "ds_symbol": "USDCHF",
                                                         "deal_size": 0.05,
                                                         "broker_stop_coefficient": 0.998,
                                                         "direction": "low-long",
                                                         "max_pivot_points": 10,
                                                         "max_bars_to_check": 100,
                                                         "dont_wait_for_confirmation": False,
                                                         "min_number_of_divergence": {
                                                             "entry": 1,
                                                             "exit_sl": 1,
                                                             "exit_tp": 1
                                                         },
                                                         "divergence_type": "Regular",
                                                         "open_config": {
                                                             "resolution": mt5.TIMEFRAME_M3,
                                                             "pivot_period": 10,
                                                             "entry": {
                                                                 "rsi": {
                                                                     "rsi_length": 14
                                                                 },
                                                             }
                                                         },
                                                         "close_config": {
                                                             "resolution": mt5.TIMEFRAME_M1,
                                                             "pivot_period": 10,
                                                             "entry": {
                                                                 "rsi": {
                                                                     "rsi_length": 14
                                                                 },
                                                             }
                                                         }}),
                     "pair_with_wrong_deal_size_params": PairConfig(broker=initialization_config,
                                                                    data_source={"name": "mt5", "connection": initialization_config},
                                                                    **{"symbol": "[SP500]",
                                                                       "ds_symbol": "[SP500]",
                                                                       "deal_size": 1000.0,
                                                                       "broker_stop_coefficient": 0.994,
                                                                       "direction": "low-long",
                                                                       "max_pivot_points": 10,
                                                                       "max_bars_to_check": 100,
                                                                       "dont_wait_for_confirmation": False,
                                                                       "min_number_of_divergence": {
                                                                           "entry": 1,
                                                                           "exit_sl": 1,
                                                                           "exit_tp": 1
                                                                       },
                                                                       "divergence_type": "Regular",
                                                                       "open_config": {
                                                                           "resolution": mt5.TIMEFRAME_M1,
                                                                           "pivot_period": 10,
                                                                           "entry": {
                                                                               "rsi": {
                                                                                   "rsi_length": 14
                                                                               },
                                                                           }
                                                                       },
                                                                       "close_config": {
                                                                           "resolution": mt5.TIMEFRAME_M1,
                                                                           "pivot_period": 10,
                                                                           "entry": {
                                                                               "rsi": {
                                                                                   "rsi_length": 14
                                                                               },
                                                                           }
                                                                       }}),
                     "pair_with_wrong_stop_params": PairConfig(broker=initialization_config,
                                                                data_source={"name": "mt5", "connection": initialization_config},
                                                               **{"symbol": "BTCUSD",
                                                                  "ds_symbol": "BTCUSD",
                                                                  "resolution": mt5.TIMEFRAME_M3,
                                                                  "deal_size": 0.05,
                                                                  "broker_stop_coefficient": 0.999,
                                                                  "direction": "low-long",
                                                                  "max_pivot_points": 10,
                                                                  "max_bars_to_check": 100,
                                                                  "dont_wait_for_confirmation": False,
                                                                  "min_number_of_divergence": {
                                                                      "entry": 1,
                                                                      "exit_sl": 1,
                                                                      "exit_tp": 1
                                                                  },
                                                                  "divergence_type": "Regular",
                                                                  "open_config": {
                                                                      "resolution": mt5.TIMEFRAME_M1,
                                                                      "pivot_period": 10,
                                                                      "entry": {
                                                                          "rsi": {
                                                                              "rsi_length": 14
                                                                          },
                                                                      }
                                                                  },
                                                                  "close_config": {
                                                                      "resolution": mt5.TIMEFRAME_M1,
                                                                      "pivot_period": 10,
                                                                      "entry": {
                                                                          "rsi": {
                                                                              "rsi_length": 14
                                                                          },
                                                                      }
                                                                  }}),
                     }
           }


@pytest.fixture(scope="function")
def make_pairs(session):
    pairs_dict = {}
    broker = session["broker"]
    for pair_key in session["pairs"]:
        pair_params = session["pairs"][pair_key]
        pair_key_short = pair_key[:-7]
        pairs_dict[pair_key_short] = BasePair(broker, pair_params)

    yield pairs_dict


@pytest.fixture(scope="function")
def buy_two_instruments(session, make_pairs):
    broker = session["broker"]
    first_pair = make_pairs["first_pair"]
    price = first_pair.get_historical_data(resolution=first_pair.open_config.resolution).iloc[-1]["close"]

    response = first_pair.create_position(price, type_action=broker.ORDER_TYPE_BUY)

    second_pair = make_pairs["second_pair"]
    price = second_pair.get_historical_data(resolution=second_pair.open_config.resolution).iloc[-1]["close"]
    response = second_pair.create_position(price, type_action=broker.ORDER_TYPE_BUY)

    open_pos = broker.positions_total()
    assert open_pos >= np.floor(first_pair.deal_size) + np.floor(second_pair.deal_size)
    yield {"first_pair": first_pair,
           "second_pair": second_pair}



def test_create_position(session, make_pairs):
    broker = session["broker"]

    # buy all pairs
    for pair in [make_pairs["first_pair"], make_pairs["second_pair"]]:

        price = pair.get_historical_data(resolution=pair.open_config.resolution).iloc[-1]["close"]

        response = pair.create_position(price, type_action=broker.ORDER_TYPE_BUY)
        assert response is not None

        try:
            assert response.order != 0
            assert response.volume == pair.deal_size
        except Exception as e:
            print(e)
        sleep(20)


def test_close_opened_position(session, make_pairs):
    broker = session["broker"]

    # sell the first pair with identifier
    pair = make_pairs["first_pair"]
    open_pos = broker.positions_total()
    assert open_pos > 0
    price = pair.get_historical_data(resolution=pair.open_config.resolution).iloc[-1]["close"]
    response = pair.close_opened_position(price,
                                          type_action=broker.ORDER_TYPE_SELL,
                                          identifiers=[broker.positions_get(symbol=pair.symbol)[0].identifier])[0]
    assert response is not None
    assert response.volume == pair.deal_size

    # sell the second pair without any identifier
    pair = make_pairs["second_pair"]
    open_pos = broker.positions_total()
    assert open_pos > 0
    price = pair.get_historical_data(resolution=pair.open_config.resolution).iloc[-1]["close"]
    response = pair.close_opened_position(price, type_action=broker.ORDER_TYPE_SELL)[0]
    assert response is not None
    assert response.volume == pair.deal_size


def test_close_first_buyed_position(session, buy_two_instruments):
    broker = session["broker"]
    first_pair = buy_two_instruments["first_pair"]
    second_pair = buy_two_instruments["second_pair"]

    sleep(20)

    price = first_pair.get_historical_data(resolution=first_pair.open_config.resolution).iloc[-1]["close"]
    response = first_pair.close_opened_position(price, type_action=broker.ORDER_TYPE_SELL)[0]
    assert response is not None
    assert np.allclose(response.volume, first_pair.deal_size)

    price = second_pair.get_historical_data(resolution=second_pair.open_config.resolution).iloc[-1]["close"]
    response = second_pair.close_opened_position(price, type_action=broker.ORDER_TYPE_SELL)[0]
    assert response is not None
    assert np.allclose(response.volume, second_pair.deal_size)


def test_close_second_buyed_position(session, buy_two_instruments):
    broker = session["broker"]
    first_pair = buy_two_instruments["first_pair"]
    second_pair = buy_two_instruments["second_pair"]

    sleep(20)

    price = second_pair.get_historical_data(resolution=second_pair.open_config.resolution).iloc[-1]["close"]
    response = second_pair.close_opened_position(price, type_action=broker.ORDER_TYPE_SELL)[0]
    assert response is not None
    assert np.allclose(response.volume, second_pair.deal_size)

    price = first_pair.get_historical_data(resolution=first_pair.open_config.resolution).iloc[-1]["close"]
    response = first_pair.close_opened_position(price, type_action=broker.ORDER_TYPE_SELL)[0]
    assert response is not None
    assert np.allclose(response.volume, first_pair.deal_size)

# #  python -m pytest .\Tests\test_open_close.py
# # --trace - option for debug
# #  s | n - next, s - step into
