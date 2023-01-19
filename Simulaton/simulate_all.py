import argparse
import os
from datetime import datetime
import pandas as pd
import multiprocessing as mp
from Simulaton.simulate import run_simulate
import json


def run_batch_simulate(tasks_file: str, out_file: str, batch_id: int) -> None:
    def concat_df(result: tuple) -> None:
        total_yield, params = result
        nonlocal df
        df = pd.concat([df, pd.DataFrame({"timestamp": datetime.now().timestamp(),
                                          "symbol": params["symbol"], "resolution": params["resolution"],
                                          "dft_period": params["dft_period"],
                                          "stop_coef": params["stop_coefficient"],
                                          "hf": params["highest_fib"], "lf": params["lowest_fib"],
                                          "down_periods": params["down_periods"], "total_yield": total_yield},
                                         index=[0])])

    def error_callback(error):
        print(error)

    try:
        with open(tasks_file) as f:
            tasks = json.load(f)
            if batch_id != -1:
                tasks = [t for t in tasks if t["batch_id"] == batch_id]

        if os.path.exists(out_file):
            df = pd.read_csv(out_file)
        else:
            df = pd.DataFrame(columns=["timestamp", "symbol", "resolution", "dft_period", "stop_coef",
                                       "hf", "lf", "down_periods", "total_yield"])

        pool = mp.Pool(mp.cpu_count())

        for task in tasks:
            pool.apply_async(run_simulate, kwds={"symbol": task["symbol"], "resolution": task["r"],
                                                 "stop_coefficient": task["st"],
                                                 "dft_period": task["d"], "down_periods": task["dp"],
                                                 "highest_fib": task["hf"], "lowest_fib": task["lf"]},
                             callback=concat_df, error_callback=error_callback)

        pool.close()
        pool.join()
        df[["timestamp", "symbol", "resolution", "dft_period", "stop_coef",
            "hf", "lf", "down_periods", "total_yield"]].to_csv(out_file, index=False)

    except FileNotFoundError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_file", "-tf", default="batched_task_list.json", help="JSON with batched task list")
    parser.add_argument("--out_file", "-o", default="simulate_result.csv", help="CSV output file")
    parser.add_argument("--batch_id", "-b", default=0, type=int, help="batch_id. If -1, all batches are simulated")

    args = parser.parse_args()

    run_batch_simulate(args.task_file, args.out_file, args.batch_id)
