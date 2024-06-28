import sys
import os
import traceback
from datetime import datetime, timezone, timedelta
import json
import logging
from pair.multiindpair import MultiIndPair
import MetaTrader5 as mt5
from time import sleep
import argparse
import multiprocessing as mp
from utils import wait_for_next_hour, sleep_with_dummy_requests
from pair.external_history import CapitalConnection
from models import BotConfig, PairConfig
from pair.enums import DataSource


def continuous_trading_pair(p_config: PairConfig, **kwargs):
    tz = timezone(timedelta(hours=1))
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        filename=f"./History/logs/trading_log_{p_config.login}_"
                                 f"{datetime.now(tz).date()}_{p_config.symbol}.txt",
                        filemode="a")

    mt5.initialize(login=p_config.login, password=p_config.password,
                   server=p_config.server, path=str(p_config.path))
    p = MultiIndPair(mt5, p_config)

    while True:
        try:
            p.update_positions()

            configs_to_check = p.get_configs_to_check()
            p.update_last_check_time(configs_to_check)

            for cnf in configs_to_check:
                p.set_parameters_by_config(cnf)
                data = p.get_historical_data(**kwargs)

                if data is None:
                    sleep(p.resolution * 60)
                    continue

                type_action, action_details = p.strategy.get_action(data=data,
                                                                    symbol=p.symbol,
                                                                    positions=p.positions,
                                                                    stop_coefficient=p.broker_stop_coefficient,
                                                                    trade_tick_size=p.trade_tick_size,
                                                                    config_type=cnf["applied_config"])

                if type_action in cnf["available_actions"]:
                    p.make_action(type_action, action_details)
                    sleep(5)
                    p.update_positions()

                elif type_action is not None:
                    print(f"{datetime.now().time().isoformat(timespec='minutes')} {p.symbol}: Action {type_action} is not available for {cnf['applied_config']} config and {len(p.positions)} positions and {p.strategy.direction} direction")

            time_to_sleep = min([p.__getattribute__(f"{cnf['applied_config']}_config").resolution for cnf in configs_to_check], default=p.min_resolution)
            print(f"{datetime.now().time().isoformat(timespec='minutes')} {p.symbol}: Sleep for {time_to_sleep} minutes")
            sleep_with_dummy_requests(time_to_sleep, p, **kwargs)
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
