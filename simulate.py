from datetime import datetime, timedelta
import pytz
import argparse
import logging
import pandas as pd
from models.multiind_models import BotConfig, PairConfig
from pair.simulate_pair import SimulatePair
import MetaTrader5 as mt5
import json


def run_simulate(balance: float = 1000.0, start_date: datetime = None):
    initial_balance = balance
    mt5.initialize(login=config.broker.login, password=config.broker.password,
                   server=config.broker.server, path=str(config.broker.path))

    pairs = []

    # receiving historical data
    historical_data_open = []
    historical_data_close = []
    for symbol, params in config.symbol_parameters.items():
        combined_data = {**config.dict(), **params, "symbol": symbol}
        pair_config: PairConfig = PairConfig.model_validate(combined_data)
        pair = SimulatePair(mt5, pair_config)
        n_days = 365 if not start_date else (datetime.now(tz=pytz.UTC) - start_date).days + 1
        open_numpoints = n_days * (24 * 60 // pair.open_config.resolution) + pair.strategy.numpoints
        close_numpoints = n_days * (24 * 60 // pair.close_config.resolution) + pair.strategy.numpoints
        pairs.append(pair)
        historical_data_open.append(pair.get_historical_data(resolution=pair.open_config.resolution,
                                                             numpoints=open_numpoints,
                                                             start_date=start_date))
        historical_data_close.append(
            pair.get_historical_data(resolution=pair.close_config.resolution,
                                     numpoints=close_numpoints,
                                     start_date=start_date))
    open_full_df = pd.concat(historical_data_open, axis=1).ffill().bfill()
    close_full_df = pd.concat(historical_data_close, axis=1).ffill().bfill()

    # simulation loop
    dt_points = list((open_full_df if pairs[0].open_config.resolution == pairs[0].min_resolution else close_full_df).index)
    df_open_points = list(open_full_df.index)

    curr_dt = df_open_points[pairs[0].strategy.numpoints]
    while curr_dt <= dt_points[-1]:
        for i, pair in enumerate(pairs):
            open_data = open_full_df.loc[:curr_dt].iloc[:, 5 * i: 5 * (i + 1)]
            close_data = close_full_df.loc[:curr_dt].iloc[:, 5 * i: 5 * (i + 1)]
            balance = pair.make_simulation_step(curr_dt, open_data, close_data, balance)
            msg = f"Step {pair.symbol}, {curr_dt}. Balance: {balance}, positions: {pair.positions}"
            logging.info(msg)
            print(msg)

        try:
            if all([not x.positions for x in pairs]) and curr_dt < dt_points[-1]:
                curr_dt = [x for x in df_open_points if x > curr_dt][0]
            else:
                curr_dt += timedelta(minutes=pairs[0].min_resolution)
        except IndexError:
            break

    for i, pair in enumerate(pairs):
        if pair.positions:
            final_price = close_full_df.iloc[-1, 5 * i: 5 * (i + 1)]["close"]
            balance += sum([x.volume * final_price * ((-1) ** int(x.type == 1))for x in pair.positions])
            msg = f"Final closing positions, {pair.symbol}, price={final_price}, balance={balance}"
            logging.info(msg)
            print(msg)
    print("History:\n", pair.history)
    msg = f"Total profit {balance - initial_balance} USD, {(balance - initial_balance) / initial_balance * 100}%"
    logging.info(msg)
    print(msg)


def valid_date(s: str):
    try:
        return datetime.strptime(s, "%d-%m-%Y").replace(tzinfo=pytz.UTC)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Not a valid date: '{s}'. Expected format: DD-MM-YYYY")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--initial_amount", type=float)
    parser.add_argument("--start_date", type=valid_date, help="The date in DD-MM-YYYY format",)
    args = parser.parse_args()

    config = json.load(open(args.config_file))
    config = BotConfig(**config)

    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        filename=f"./History/logs/simulate_log_{datetime.now().date()}.txt",
                        filemode="w")

    run_simulate(balance=args.initial_amount, start_date=args.start_date)
