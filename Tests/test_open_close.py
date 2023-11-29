import json
import numpy as np
import pytest
import MetaTrader5 as mt5
from pair.multiindpair import MultiIndPair
from time import sleep


@pytest.fixture(scope="module")
def session():
    config = json.load(open("multi_config_shares.json"))
    mt5.initialize(login=config["login"], password=config["password"],
                   server=config["server"], path=config["path"])
    yield {"broker": mt5,
           "pairs": {"first_pair_params": {"symbol": "USDCAD",
                                           "data_source": "mt5",
                                           "deal_size": 0.05,
                                           "stop_coefficient": 0.997,
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
                                           }},
                     "second_pair_params": {"symbol": "USDCHF",
                                            "data_source": "mt5",
                                            "deal_size": 0.05,
                                            "stop_coefficient": 0.998,
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
                                            }},
                     "pair_with_wrong_deal_size_params": {"symbol": "[SP500]",
                                                          "data_source": "mt5",
                                                          "deal_size": 1000.0,
                                                          "stop_coefficient": 0.994,
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
                                                          }},
                     "pair_with_wrong_stop_params": {"symbol": "BTCUSD",
                                                     "data_source": "mt5",
                                                     "resolution": mt5.TIMEFRAME_M3,
                                                     "deal_size": 0.05,
                                                     "stop_coefficient": 0.999,
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
                                                     }},
                     }
           }


@pytest.fixture(scope="function")
def make_pairs(session):
    pairs_dict = {}
    broker = session["broker"]
    for pair_key in session["pairs"]:
        pair_params = session["pairs"][pair_key]
        pair_key_short = pair_key[:-7]
        pairs_dict[pair_key_short] = MultiIndPair(broker, 0,  pair_params)

    yield pairs_dict


@pytest.fixture(scope="function")
def buy_two_instruments(session, make_pairs):
    broker = session["broker"]
    first_pair = make_pairs["first_pair"]
    price = first_pair.get_historical_data().iloc[-1]["close"]

    response = first_pair.create_position(price, type_action=broker.ORDER_TYPE_BUY)

    second_pair = make_pairs["second_pair"]
    price = second_pair.get_historical_data().iloc[-1]["close"]
    response = second_pair.create_position(price, type_action=broker.ORDER_TYPE_BUY)

    open_pos = broker.positions_total()
    assert open_pos >= np.floor(first_pair.deal_size) + np.floor(second_pair.deal_size)
    yield {"first_pair": first_pair,
           "second_pair": second_pair}


# def test_check_order(session, make_pairs):
#     broker = session["broker"]
#
#     wrong_pairs = (make_pairs["pair_with_wrong_deal_size"],
#                    make_pairs["pair_with_wrong_stop"])
#
#     assert make_pairs["pair_with_wrong_deal_size"] is not None
#     assert make_pairs["pair_with_wrong_stop"] is not None
#
#     for pair in wrong_pairs:
#         price = pair.get_curr_price()
#
#         response = pair.create_position(price, type_action=broker.ORDER_TYPE_BUY)
#         assert response is None


def test_create_position(session, make_pairs):
    broker = session["broker"]

    # buy all pairs
    for pair in [make_pairs["first_pair"], make_pairs["second_pair"]]:

        price = pair.get_historical_data().iloc[-1]["close"]

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
    price = pair.get_historical_data().iloc[-1]["close"]
    response = pair.close_opened_position(price,
                                          type_action=broker.ORDER_TYPE_SELL,
                                          identifiers=[broker.positions_get(symbol=pair.symbol)[0].identifier])[0]
    assert response is not None
    assert response.volume == pair.deal_size

    # sell the second pair without any identifier
    pair = make_pairs["second_pair"]
    open_pos = broker.positions_total()
    assert open_pos > 0
    price = pair.get_historical_data().iloc[-1]["close"]
    response = pair.close_opened_position(price, type_action=broker.ORDER_TYPE_SELL)[0]
    assert response is not None
    assert response.volume == pair.deal_size


def test_close_first_buyed_position(session, buy_two_instruments):
    broker = session["broker"]
    first_pair = buy_two_instruments["first_pair"]
    second_pair = buy_two_instruments["second_pair"]

    sleep(20)

    price = first_pair.get_historical_data().iloc[-1]["close"]
    response = first_pair.close_opened_position(price, type_action=broker.ORDER_TYPE_SELL)[0]
    assert response is not None
    assert np.allclose(response.volume, first_pair.deal_size)

    price = second_pair.get_historical_data().iloc[-1]["close"]
    response = second_pair.close_opened_position(price, type_action=broker.ORDER_TYPE_SELL)[0]
    assert response is not None
    assert np.allclose(response.volume, second_pair.deal_size)


def test_close_second_buyed_position(session, buy_two_instruments):
    broker = session["broker"]
    first_pair = buy_two_instruments["first_pair"]
    second_pair = buy_two_instruments["second_pair"]

    sleep(20)

    price = second_pair.get_historical_data().iloc[-1]["close"]
    response = second_pair.close_opened_position(price, type_action=broker.ORDER_TYPE_SELL)[0]
    assert response is not None
    assert np.allclose(response.volume, second_pair.deal_size)

    price = first_pair.get_historical_data().iloc[-1]["close"]
    response = first_pair.close_opened_position(price,  type_action=broker.ORDER_TYPE_SELL)[0]
    assert response is not None
    assert np.allclose(response.volume, first_pair.deal_size)

# #  python -m pytest .\Tests\test_open_close.py
# # --trace - option for debug
# #  s | n - next, s - step into
