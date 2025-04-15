import json
import pytest
import MetaTrader5 as mt5
from pair.multiindpair import BasePair
from models.multiind_models import BotConfig, PairConfig
from models.base_models import BasePairConfig


@pytest.fixture(scope="module")
def session():
    config = json.load(open("../42503701a-adm.json"))
    config = BotConfig(**config)
    initialization_config = {"login": config.broker.login,
                             "password": config.broker.password,
                             "server": config.broker.server,
                             "path": str(config.broker.path)}
    alarm_config = config.alarm_config
    mt5.initialize(**initialization_config)
    yield {"broker": mt5,
           "pairs": {"first_pair_params": PairConfig(broker=initialization_config,
                                                     data_source={"name": "mt5", "connection": initialization_config},
                                                     action_methods=["alarm"],
                                                     alarm_config=alarm_config,
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
                                                        }})
           }
           }


@pytest.fixture(scope="function")
def make_pairs(session):
    pairs_dict = {}
    broker = session["broker"]
    for pair_key in session["pairs"]:
        pair_params = session["pairs"][pair_key]
        pair_key_short = pair_key[:-7]
        pairs_dict[pair_key_short] = BasePair(broker, BasePairConfig.model_validate(pair_params.dict()))
    yield pairs_dict


def test_send_email(session, make_pairs):
    for _, pair in make_pairs.items():
        pair.send_email("savahome@list.ru", "Test e-mail", "Hi, User!")


# #  python -m pytest .\Tests\test_open_close.py
# # --trace - option for debug
# #  s | n - next, s - step into
