import os
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


def continuous_trading_pair(p_config: dict, **kwargs):
    tz = timezone(timedelta(hours=1))
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        filename=f"./History/logs/trading_log_{p_config['login']}_"
                                 f"{datetime.now(tz).date()}_{p_config['symbol']}.txt",
                        filemode="a")

    mt5.initialize(login=p_config['login'], password=p_config['password'],
                   server=p_config['server'], path=p_config['path'])
    p = MultiIndPair(mt5, p_config)

    while True:

        broker_positions = p.broker.positions_get(symbol=p.symbol)
        p.positions = broker_positions if broker_positions is not None else []
        p.set_parameters_by_position()

        data = p.get_historical_data(**kwargs)

        if data is None:
            sleep(p.resolution * 60)
            continue

        type_action, action_details = p.strategy.get_action(data, p)

        p.make_action(type_action, action_details)

        print(f"{p.symbol}: Sleep for {p.resolution} minutes")
        sleep_with_dummy_requests(p, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    parser.add_argument("--immediately", "-i", action="store_true")
    args = parser.parse_args()

    config = json.load(open(args.config_file))

    n_symbols = len(config["symbols_parameters"])
    pool = mp.Pool(n_symbols)

    if not args.immediately:
        wait_for_next_hour(verbose=True)

    try:

        kws = {}

        if config["data_source"] == "capital":
            api_key, identifier, password = config["capital_creds"].values()
            capital_conn = CapitalConnection(api_key=api_key, identifier=identifier, password=password)
            kws.update({"capital_conn": capital_conn})

        for symbol, params in config["symbols_parameters"].items():
            pair_config = {k: v for k, v in config.items() if k != "symbols_parameters"}
            pair_config.update({"symbol": symbol,
                                "datasource_symbol": params["ds_symbol"],
                                "deal_size": params["deal_size"],
                                "direction": params["direction"]})

            pool.apply_async(continuous_trading_pair, args=(pair_config,), kwds=kws,
                             error_callback=print)

    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt")
        pool.terminate()

    else:
        pool.close()

    pool.join()
