import argparse
import os
from datetime import datetime
import pandas as pd
import multiprocessing as mp
from simulation.simulate import run_simulate
import json


def run_batch_simulate(tasks_file: str, out_file: str, batch_id: int,
                       max_neg_inrow: int, min_total_yield: float) -> None:
    def concat_df(result: tuple) -> None:
        try:
            total_yield, params = result
            nonlocal df
            df = pd.concat([df, pd.DataFrame({"timestamp": datetime.now().timestamp(),
                                              "symbol": params["symbol"], "resolution": params["resolution"],
                                              "dft_period": params["dft_period"],
                                              "stop_coef": str(params["stop_coefficient"]),
                                              "hf": params["highest_fib"], "lf": params["lowest_fib"],
                                              "down_periods": str(params["down_periods"]),
                                              "is_multibuying_available": params["is_multibuying_available"],
                                              "upper_timeframe_parameters": str(params["upper_timeframe_parameters"]),
                                              "unclear_trend_periods": params["unclear_trend_periods"],
                                              "down_trend_periods": params["down_trend_periods"],
                                             "total_yield": total_yield},
                                             index=[0])])
            df[["timestamp", "symbol", "resolution", "dft_period", "stop_coef",
                "hf", "lf", "down_periods", "upper_timeframe_parameters",
                "unclear_trend_periods", "down_trend_periods",
                "total_yield"]].to_csv(out_file, index=False)

        except Exception as error:
            print(f"Error in concat_df(), {error}")

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
                                       "hf", "lf", "down_periods", "upper_timeframe_parameters",
                                       "unclear_trend_periods", "down_trend_periods",
                                       "total_yield"])
        max_cpu = 8
        n_cpu = min(len(tasks), mp.cpu_count(), max_cpu)
        pool = mp.Pool(n_cpu)

        for task in tasks:
            pool.apply_async(run_simulate, kwds={"symbol": task["symbol"],
                                                 "yahoo_symbol": task["symbol"], "resolution": task["r"],
                                                 "stop_coefficient": task["st"],
                                                 "dft_period": task["d"], "down_periods": task["dp"],
                                                 "highest_fib": task["hf"], "lowest_fib": task["lf"],
                                                 "is_multibuying_available": False,
                                                 "upper_timeframe_parameters": task["upper_timeframe_parameters"],
                                                 "unclear_trend_periods": task["unclear_trend_periods"],
                                                 "down_trend_periods": task["down_trend_periods"],
                                                 "max_neg_inrow": max_neg_inrow,
                                                 "min_total_yield": min_total_yield},
                             callback=concat_df, error_callback=error_callback)

        pool.close()
        pool.join()

    except FileNotFoundError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_file", "-tf", default="batched_task_list.json", help="JSON with batched task list")
    parser.add_argument("--out_file", "-o", default="simulate_result.csv", help="CSV output file")
    parser.add_argument("--batch_id", "-b", default=0, type=int, help="batch_id. If -1, all batches are simulated")
    parser.add_argument("-n", "--max_neg_inrow", type=int, default=3,
                        help="Max negative trades in a row to break simulation")
    parser.add_argument("-y", "--min_total_yield", type=float, default=-0.5,
                        help="Min percentage of total yeild to break simulation")

    args = parser.parse_args()

    run_batch_simulate(args.task_file, args.out_file, args.batch_id, args.max_neg_inrow, args.min_total_yield)
