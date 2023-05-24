import pytest
from pair import Pair
import pandas as pd
from pair.check_data import fix_missing, fix_outliers_and_missing


@pytest.fixture(scope="function")
def raw_pair():
    raw_pair = Pair(symbol="[SP500]",
                    resolution=3,
                    deal_size=1.0,
                    stop_coef=0.994,
                    dft_period=19)

    raw_pair.prices = pd.read_csv("../History/dft_2022-09-05_3min.csv", usecols=[0, 1, 2, 3, 4], index_col=0)
    yield raw_pair


def test_fix_missings(raw_pair):
    n_rows = int(raw_pair.prices.shape[0] * 0.7)
#    full_df = raw_pair.prices[["close", "open", "high", "low"]]
    has_missings, full_df = fix_missing(raw_pair.prices,
                                        ["close", "open", "high", "low"],
                                        n_rows)
    assert full_df[-n_rows:].isna().sum().sum() == 0


def test_fix_outliers_and_missings(raw_pair):
    n_rows = int(raw_pair.prices.shape[0] * 0.7)
    full_df = fix_outliers_and_missing(raw_pair.prices,
                                       ["close", "open", "high", "low"],
                                       n_rows)
    assert full_df[-n_rows:].isna().sum().sum() == 0
