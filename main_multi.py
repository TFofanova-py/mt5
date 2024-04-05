import os
import sys
import traceback
from datetime import datetime, timezone, timedelta
import json
import logging
import pandas as pd
from pair.multiindpair import MultiIndPair
import MetaTrader5 as mt5
from time import sleep
import argparse
import multiprocessing as mp
from utils import wait_for_next_hour, sleep_with_dummy_requests
from pair.external_history import CapitalConnection


def continuous_trading_pair(p_config: dict, strategy_id: int, **kwargs):
    tz = timezone(timedelta(hours=1))
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        filename=f"./History/logs/trading_log_{p_config['login']}_"
                                 f"{datetime.now(tz).date()}_{p_config['symbol']}.txt",
                        filemode="a")

    mt5.initialize(login=p_config['login'], password=p_config['password'],
                   server=p_config['server'], path=p_config['path'])
    p = MultiIndPair(mt5, strategy_id, p_config)

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

                type_action, action_details = p.strategy.get_action(data, p, config_type=cnf["applied_config"])

                if type_action in cnf["available_actions"]:
                    p.make_action(type_action, action_details)
                    sleep(5)
                    p.update_positions()

                elif type_action is not None:
                    print(f"{datetime.now().time().isoformat(timespec='minutes')} {p.symbol}: Action {type_action} is not available for {cnf['applied_config']} config and {len(p.positions)} positions and {p.strategy.direction} direction")

            time_to_sleep = min([p.strategy.resolution_set[cnf["applied_config"]] for cnf in configs_to_check], default=p.min_resolution)
            print(f"{datetime.now().time().isoformat(timespec='minutes')} {p.symbol}: Sleep for {time_to_sleep} minutes")
            sleep_with_dummy_requests(time_to_sleep, p, **kwargs)
        except:
            raise Exception("".join(traceback.format_exception(*sys.exc_info())))


def save_strategy(config: dict, file_name: str = "strategies.csv") -> int:

    df_strategy = None
    if os.path.exists(file_name):
        df_strategy = pd.read_csv(file_name)

    has_indicator = {k: config.get("entry", {}).get(k) is not None for k in
                    ["rsi", "macd", "momentum", "cci", "obv", "stk",
                     "vwmacd", "cmf", "mfi"]}

    data = pd.DataFrame.from_dict({"direction": [config.get("direction")],
                                   "resolution_open": [config["resolution"]["open"]],
                                   "resolution_close": [config["resolution"]["close"]],
                                   "stop_coefficient": [config["stop_coefficient"]],
                                   "pivot_open": [config["pivot_period"]["open"]],
                                   "pivot_close": [config["pivot_period"]["close"]],
                                   "n_divergence_entry": [config["min_number_of_divergence"]["entry"]],
                                   "n_divergence_sl": [config["min_number_of_divergence"]["exit_sl"]],
                                   "n_divergence_tp": [config["min_number_of_divergence"]["exit_tp"]],
                                   "max_pivot_points": [config.get("max_pivot_points")],
                                   "max_bars_to_check": [config.get("max_bars_to_check")],
                                   "dont_wait_for_confirmation": [config["dont_wait_for_confirmation"]],
                                   "has_rsi": [has_indicator["rsi"]],
                                   "rsi_length": [config.get("entry", {}).get("rsi", {}).get("rsi_length")],
                                   "has_macd": [has_indicator["macd"]],
                                   "macd_fast": [config.get("entry", {}).get("macd", {}).get("fast_length")],
                                   "macd_slow": [config.get("entry", {}).get("macd", {}).get("slow_length")],
                                   "macd_signal": [config.get("entry", {}).get("macd", {}).get("signal_length")],
                                   "has_momentum": [has_indicator["momentum"]],
                                   "momentum_length": [config.get("entry", {}).get("momentum", {}).get("length")],
                                   "has_cci": [has_indicator["cci"]],
                                   "cci_length": [config.get("entry", {}).get("cci", {}).get("length")],
                                   "has_obv": [has_indicator["obv"]],
                                   "has_stk": [has_indicator["stk"]],
                                   "stk_stoch": [config.get("entry", {}).get("stk", {}).get("stoch_length")],
                                   "stk_sma": [config.get("entry", {}).get("stk", {}).get("sma_length")],
                                   "vwmacd": [has_indicator["vwmacd"]],
                                   "vwmacd_fast": [config.get("entry", {}).get("vwmacd", {}).get("fast_length")],
                                   "vwmacd_slow": [config.get("entry", {}).get("vwmacd", {}).get("slow_length")],
                                   "vwmacd_signal": [config.get("entry", {}).get("vwmacd", {}).get("signal_length")],
                                   "has_cmf": [has_indicator["cmf"]],
                                   "cmf_length": [config.get("entry", {}).get("cmf", {}).get("length")],
                                   "has_mfi": [has_indicator["mfi"]],
                                   "mfi_length": [config.get("entry", {}).get("mfi", {}).get("length")]
                                   })

    if df_strategy is None:
        data["id"] = 0
        df_strategy = data

    else:
        df_ext = pd.concat([df_strategy.drop("id", axis=1), data])

        if df_ext.duplicated().iloc[-1]:
            return int(df_strategy[df_ext.duplicated(keep="last").iloc[:-1]].iloc[0]["id"])

        else:
            data["id"] = df_strategy["id"].max() + 1
            df_strategy = pd.concat([df_strategy, data])

    df_strategy.to_csv(file_name, index=False)
    return int(data["id"])


def print_error(er):
    if str(er) != "KeyboardInterrupt":
        print(er)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    parser.add_argument("--immediately", "-i", action="store_true")
    args = parser.parse_args()

    config = json.load(open(args.config_file))
    # strategy_id = save_strategy(config)
    strategy_id = 0

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

            pool.apply_async(continuous_trading_pair, args=(pair_config, strategy_id), kwds=kws,
                             error_callback=print_error)

    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt")
        pool.terminate()

    else:
        pool.close()

    pool.join()
