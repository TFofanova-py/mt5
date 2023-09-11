from datetime import datetime, timezone, timedelta
import json
import logging
from pair.multiindpair import MultiIndPair
import MetaTrader5 as mt5
from time import sleep
import numpy as np
import argparse
import multiprocessing as mp
from utils import wait_for_next_hour


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
        p.set_parameters_by_position()

        data = p.get_historical_data()

        if data is None:
            sleep(p.resolution * 60)
            continue

        divergences_cnt = p.count_divergence(data)
        if divergences_cnt["top"] + divergences_cnt["bottom"] > 0:
            print(f"{p.symbol}: divergences {divergences_cnt}")

        type_action = None
        if p.direction in ["low-long", "bi"] and \
                divergences_cnt["bottom"] >= p.min_number_of_divergence["entry"] and \
                len(p.positions) == 0:
            type_action = mt5.ORDER_TYPE_BUY

        elif p.direction in ["high-short", "bi"] and \
                divergences_cnt["top"] >= p.min_number_of_divergence["entry"] and \
                len(p.positions) == 0:
            type_action = mt5.ORDER_TYPE_SELL

        if type_action is not None:
            price = data["close"].iloc[-1]
            response = p.create_position(price=price, type_action=type_action)

            logging.info(f"{data.index[-1]}, open position: {response}")
            print(response)

        elif len(p.positions) > 0:
            same_direction_divergences = divergences_cnt["bottom"] if p.positions[0].type == 0 \
                else divergences_cnt["top"]
            opposite_direction_divergences = divergences_cnt["top"] if p.positions[0].type == 0 else divergences_cnt[
                "bottom"]

            if same_direction_divergences >= p.min_number_of_divergence["exit_sl"] or \
                    opposite_direction_divergences >= p.min_number_of_divergence["exit_tp"]:
                # close position
                price = data["close"].iloc[-1]
                type_action = (p.positions[0].type + 1) % 2
                responses = p.close_opened_position(price=price, type_action=type_action)
                print(responses)
                logging.info(f"{data.index[-1]}, close position: {responses}")

                swing = opposite_direction_divergences >= p.min_number_of_divergence["entry"]
                if p.direction == "bi" and swing and p.resolution == p.resolution_set["open"]:
                    # open next position in opposite direction
                    response = p.create_position(price=price, type_action=type_action)
                    print(response)
                    logging.info(f"{data.index[-1]}, open position: {response}")
            else:
                # modify stop-loss
                price = data["close"].iloc[-1]
                price_goes_up = p.was_price_goes_up(data)
                new_sls = None

                if p.positions[0].type == 0 and price_goes_up:
                    # if long and price goes up, move sl up
                    new_sl = round(price * p.stop_coefficient, abs(int(np.log10(p.trade_tick_size))))
                    if new_sl > p.positions[0].sl:
                        new_sls = [new_sl]
                elif p.positions[0].type == 1 and not price_goes_up:
                    # if short and price goes down, move sl down
                    new_sl = round(price * (2. - p.stop_coefficient), abs(int(np.log10(p.trade_tick_size))))
                    if new_sl < p.positions[0].sl:
                        new_sls = [new_sl]

                if new_sls is not None:
                    response = p.modify_sl(new_sls)
                    print("modify stop-loss", response)
                    logging.info(f"{data.index[-1]}, modify position: {response}")

        if type_action is not None:
            logging.info(f"{data.index[-1]}, number divergences: {divergences_cnt}")

        print(f"{p.symbol}: Sleep for {p.resolution} minutes")
        sleep(p.resolution * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str)
    parser.add_argument("--immediately", "-i", action="store_true")
    args = parser.parse_args()

    config = json.load(open(args.config_file))

    n_symbols = len(config["symbols_and_dealsizes"])
    pool = mp.Pool(n_symbols)

    if not args.immediately:
        wait_for_next_hour(verbose=True)

    try:

        for symbol, deal_size in config["symbols_and_dealsizes"].items():
            pair_config = {k: v for k, v in config.items() if k != "symbols_and_dealsizes"}
            pair_config.update({"symbol": symbol, "deal_size": deal_size})

            pool.apply_async(continuous_trading_pair, args=(pair_config,), error_callback=print)

    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt")
        pool.terminate()

    else:
        pool.close()

    pool.join()
