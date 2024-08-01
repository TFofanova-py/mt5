import sys
import os
import traceback
from datetime import datetime, timezone, timedelta
import json
import logging
from pair.multiindpair import MultiIndPair
import MetaTrader5 as mt5
import argparse
import multiprocessing as mp
from utils import wait_for_next_hour, sleep_with_dummy_requests
from pair.external_history import CapitalConnection
from models.multiind_models import BotConfig, PairConfig
from pair.enums import DataSource


def continuous_trading_pair(p_config: PairConfig, **kwargs):
    tz = timezone(timedelta(hours=1))
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        filename=f"./History/logs/trading_log_{p_config.broker.login}_"
                                 f"{datetime.now(tz).date()}_{p_config.symbol}.txt",
                        filemode="a")

    mt5.initialize(login=p_config.broker.login, password=p_config.broker.password,
                   server=p_config.broker.server, path=str(p_config.broker.path))
    print("Connection to the broker", p_config.symbol, mt5.last_error())
    p = MultiIndPair(mt5, p_config)

    while True:
        try:
            response = p.make_trading_step()
            if not response.is_success:
                print(f"{p.symbol}, Something went wrong in making trading step")
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
    parser.add_argument("--immediately", "-i", action="store_true")
    args = parser.parse_args()

    config = json.load(open(args.config_file))
    config = BotConfig(**config)

    n_symbols = len(config.symbol_parameters)
    pool = mp.Pool(n_symbols)

    if not args.immediately:
        wait_for_next_hour(verbose=True)

    try:

        kws = {}

        if config.data_source == DataSource.capital:
            api_key, identifier, password = config.capital_creds.values()
            capital_conn = CapitalConnection(api_key=api_key, identifier=identifier, password=password)
            kws.update({"capital_conn": capital_conn})

        for symbol, params in config.symbol_parameters.items():
            combined_data = {**config.dict(), **params, "symbol": symbol}
            pair_config: PairConfig = PairConfig.model_validate(combined_data)

            pool.apply_async(continuous_trading_pair, args=(pair_config,), kwds=kws,
                             error_callback=print_error)

    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt")
        pool.terminate()

    else:
        pool.close()

    pool.join()
