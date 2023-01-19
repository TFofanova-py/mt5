import pytest
import MetaTrader5 as mt5
from config_mt5 import Config
from pair import Pair
from time import sleep
from test_open_close import make_pairs


@pytest.fixture(scope="module")
def session():
    mt5.initialize(login=Config.login, password=Config.password, server=Config.server, path=Config.path)
    yield {"broker": mt5,
           "pairs": {
               "res1_params": {"symbol": "[SP500]",
                               "resolution": 1,
                               "deal_size": 1.0,
                               "stop_coefficient": 0.994,
                               "limit": None,
                               "dft_period": 19},
               "res3_params": {"symbol": "BTCUSD",
                               "resolution": 3,
                               "deal_size": 0.5,
                               "stop_coefficient": 0.99,
                               "limit": None,
                               "dft_period": 34},
               "res6_params": {"symbol": "BTCUSD",
                               "resolution": 6,
                               "deal_size": 0.5,
                               "stop_coefficient": 0.99,
                               "limit": None,
                               "dft_period": 34}
           }
           }


def test_apply_resolution(session, make_pairs):
    broker = session["broker"]

    for pair_key in make_pairs:
        pair = make_pairs[pair_key]
        pair.fetch_prices(broker)
        result = pair.apply_resolution()

        print(f"{pair_key}, {pair.resolution}, {result.index[:5]}")
        assert result.shape[0] > 0

# #  python -m pytest test_resolution.py
# # --trace - option for debug
# #  e | n - next, s - step into
