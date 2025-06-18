import json
from time import sleep
from copy import copy
import MetaTrader5 as mt5
import pandas as pd
from pair.multiindpair import MultiIndPair
from models.multiind_models import BotConfig, PairConfig
from pair.ta_utils import analyze_ichimoku

CONFIG_FILE = "../test-ichimoku-same.json"


def stable_percents(df1, df2):
    df_concat = pd.concat([df1["close"], df2["close"]])
    df_duplicate = df_concat.duplicated(keep=False)
    return df_duplicate.sum()/df_duplicate.shape[0]


def main():
    prev_data =None
    prev_result = None

    while True:
        config = BotConfig.model_validate(json.load(open(CONFIG_FILE)))
        mt5.initialize(login=config.broker.login, password=config.broker.password,
                       server=config.broker.server, path=str(config.broker.path))

        symbol = "XAUUSD"
        params = config.symbol_parameters.get(symbol)

        combined_data = {**config.model_dump(mode="json"), **params, "symbol": symbol}
        pair_config: PairConfig = PairConfig.model_validate(combined_data)

        pair = MultiIndPair(mt5, pair_config)

        data = pair.get_historical_data(resolution=pair.open_config.resolution)  # 1, for close config - the same resolution
        if prev_data is not None:
            print("Data stability", data.index[-1], stable_percents(prev_data, data))
        prev_data = copy(data)

        tf = pair.strategy.ichimoku.layers[0].tf  # minutes, 4 hours
        resampled_data = data.resample(f"{tf}Min").last()
        last_resampled = resampled_data.index[-1]
        shift_periods = data[data.index > last_resampled].shape[0]
        resampled_data = data.shift(-shift_periods).resample(f"{tf}Min").first()

        result = analyze_ichimoku(resampled_data, periods=pair.strategy.ichimoku.periods)
        if prev_result is not None:
            print("Trend stability", data.index[-1], prev_result[0] == result[0], prev_result[0], result[0])
        prev_result = result
        sleep(60)


if __name__ == "__main__":
    main()