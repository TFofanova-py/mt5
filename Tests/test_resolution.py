import pytest
import MetaTrader5 as mt5
from config_mt5 import Config


@pytest.fixture(scope="module")
def session():
    mt5.initialize(login=Config.login, password=Config.password, server=Config.server, path=Config.path)
    yield {"broker": mt5,
           "pairs": {
               "res1_params": {"symbol": "[SP500]",
                               "data_source": "mt5",
                               "resolution": 1,
                               "deal_size": 1.0,
                               "stop_coefficient": 0.994,
                               "dft_period": 19},
               "res3_params": {"symbol": "BTCUSD",
                               "data_source": "mt5",
                               "resolution": 3,
                               "deal_size": 0.5,
                               "stop_coefficient": 0.99,
                               "dft_period": 34},
               "res6_params": {"symbol": "BTCUSD",
                               "yahoo_symbol": "BTC-USD",
                               "data_source": "yahoo",
                               "resolution": 6,
                               "deal_size": 0.5,
                               "stop_coefficient": 0.99,
                               "dft_period": 34}
           }
           }


def test_apply_resolution(session, make_pairs):
    broker = session["broker"]

    for pair_key in make_pairs:
        pair = make_pairs[pair_key]
        pair.fetch_prices(broker)
        assert pair.prices is not None, "prices is None"
        result = pair.apply_resolution(pair.prices, pair.resolution, interval="minute")

        assert result is not None and result.shape[0] > 0, "result is None"
        print(f"{pair_key}, {pair.resolution}, {result.index[:5]}")

# #  python -m pytest test_resolution.py
# # --trace - option for debug
# #  e | n - next, s - step into
