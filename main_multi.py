from datetime import datetime, timezone, timedelta
import json
import logging
from pair.multiindpair import MultiIndPair
import MetaTrader5 as mt5
from time import sleep
import numpy as np
import argparse
import multiprocessing as mp


def aggregate_signals(ar: list, threshold: int = 3) -> int:
    threshold = min(len(ar), threshold)
    n_buy_signals = (np.array(ar) == 1).astype(int).sum()
    n_sell_signals = (np.array(ar) == -1).astype(int).sum()
    return 1 if n_buy_signals >= threshold else (-1 if n_sell_signals >= threshold else 0)


def continuous_trading_pair(p_config: dict):
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
        p.positions = p.broker.positions_get(symbol=p.symbol)

        data = p.get_historical_data()

        if data is None:
            continue

        signal = p.get_divergence_signal(data)

        type_action = None
        if p.direction in ["low-long", "bi"] and signal == 1 and len(p.positions) == 0:
            type_action = mt5.ORDER_TYPE_BUY
        elif p.direction in ["high-short", "bi"] and signal == -1 and len(p.positions) == 0:
            type_action = mt5.ORDER_TYPE_SELL

        if type_action is not None:
            price = data["close"].iloc[-1]
            response = p.create_position(price=price, type_action=type_action)

            print(response)

        elif len(p.positions) > 0:
            if signal != 0:
                # close position
                price = data["close"].iloc[-1]
                type_action = (p.positions[0].type + 1) % 2
                responses = p.close_opened_position(price=price, type_action=type_action)
                print(responses)

                swing = (signal == -1 and type_action == 1) or (signal == 1 and type_action == 0)
                if p.direction == "bi" and swing:
                    type_action = type_action % 2
                    response = p.create_position(price=price, type_action=type_action)
                    print(response)
            else:
                # modify stop-loss
                price = data["close"].iloc[-1]
                price_goes_up = p.was_price_goes_up(data)
                new_sls = None

                if p.positions[0].type == 0 and price_goes_up:
                    # if long and price goes up, move sl up
                    new_sl = round(price * p.stop_coefficient, abs(int(np.log10(p.trade_tick_size))))
                    if new_sl > p.positions[0].sl:
                        new_sls = [new_sl,]
                elif p.positions[0].type == 1 and not price_goes_up:
                    # if short and price goes down, move sl down
                    new_sl = round(price * (2. - p.stop_coefficient), abs(int(np.log10(p.trade_tick_size))))
                    if new_sl < p.positions[0].sl:
                        new_sls = [new_sl,]

                if new_sls is not None:
                    response = p.modify_sl(new_sls)
                    print("modify stop-loss", response)

        print(f"{p.symbol}: Sleep for {p.resolution} minutes")
        sleep(p.resolution * 60)


def error_callback(er):
    print(er)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    args = parser.parse_args()

    config = json.load(open(args.config_file))

    n_symbols = len(config["symbols_and_dealsizes"])
    pool = mp.Pool(n_symbols)

    try:

        for symbol, deal_size in config["symbols_and_dealsizes"].items():
            pair_config = {k: v for k, v in config.items() if k != "symbols_and_dealsizes"}
            pair_config.update({"symbol": symbol, "deal_size": deal_size})

            pool.apply_async(continuous_trading_pair, args=(pair_config,), error_callback=error_callback)

    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt")
        pool.terminate()

    else:
        pool.close()

    pool.join()
