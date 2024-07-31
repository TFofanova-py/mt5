import sys
import os
import traceback
from datetime import datetime, timezone, timedelta
import json
import logging
from pair.multiindpair import BasePair
import MetaTrader5 as mt5
from time import sleep
import argparse
import multiprocessing as mp
from utils import sleep_with_dummy_requests
from pair.external_history import CapitalConnection
from models.base_models import BasePairConfig
from models.vix_models import BotConfig, PairConfig
from pair.enums import DataSource


def continuous_trading_pair(p_config: PairConfig, **kwargs):
    tz = timezone(timedelta(hours=1))
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        filename=f"./History/logs/trading_log_{p_config.broker.login}_"
                                 f"{datetime.now(tz).date()}_{p_config.ticker_to_trade}.txt",
                        filemode="a")

    mt5.initialize(login=p_config.broker.login, password=p_config.broker.password,
                   server=p_config.broker.server, path=str(p_config.broker.path))
    print("Connection to the broker", p_config.ticker_to_trade, mt5.last_error())
    casted_open_config = p_config.open_config.dict()
    casted_open_config.update({"resolution": p_config.open_config.candle_minutes})
    combined_data = {**p_config.dict(),
                     "symbol": p_config.ticker_to_trade,
                     "ds_symbol": p_config.ticker_to_monitor,
                     "open_config": casted_open_config}
    p = BasePair(mt5, BasePairConfig(**combined_data))

    while True:
        try:
            response = p.make_trading_step()
            if not response.is_success:
                print(f"{p}, something went wrong")
            sleep_with_dummy_requests(response.time_to_sleep, p, **kwargs)
        except:
            raise Exception("".join(traceback.format_exception(*sys.exc_info())))


def print_error(er):
    if str(er) != "KeyboardInterrupt":
        print(er)


if __name__ == "__main__":
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    args = parser.parse_args()

    config = json.load(open(args.config_file))
    config = BotConfig(**config)

    n_proc = len(config.trade_configs)
    pool = mp.Pool(n_proc)

    try:

        kws = {}

        if config.data_source == DataSource.capital:
            api_key, identifier, password = config.capital_creds.values()
            capital_conn = CapitalConnection(api_key=api_key, identifier=identifier, password=password)
            kws.update({"capital_conn": capital_conn})

        for trade_config in config.trade_configs:
            combined_data = {**config.dict(), **trade_config.dict()}
            pair_config: PairConfig = PairConfig.model_validate(combined_data)

            pool.apply_async(continuous_trading_pair, args=(pair_config,), kwds=kws,
                             error_callback=print_error)

    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt")
        pool.terminate()

    else:
        pool.close()

    pool.join()
