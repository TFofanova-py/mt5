import json
import argparse
from itertools import product
from pair.multiindpair import parse_du


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configurations", "-c", default="configurations.csv",
                        help="CSV file with list of configurations")
    parser.add_argument("--symbols", "-s", default="symbols_list.csv", help="CSV file with list of symbols")
    parser.add_argument("--out_file", "-o", default="batched_task_list.json", help="output JSON file")
    parser.add_argument("--batch_size", "-b", default=10, type=int, help="batch size for tasks")

    args = parser.parse_args()

    # make configuration list
    configs = []

    with open(args.configurations) as f:
        lines = f.readlines()
        # columns = lines[0].split()

        for line in lines[1:]:
            params_list = line.split(",")
            assert len(params_list) == 8, "number of parameters must be equal 8"
            r, st, d, hf, lf, dp, m, du = params_list

            configs.append({"r": int(r), "st": round(int(st) * 1e-3, 3), "d": int(d),
                            "hf": round(int(hf) * 1e-3, 3), "lf": round(int(lf) * 1e-3, 3),
                            "dp": "[" + dp + "]", "m": m == "True",
                            "du": parse_du(du)})

    # make symbols list
    with open(args.symbols) as f:
        symbols = [x.strip() for x in f.readlines()]

    # make product of them
    tasks = []
    for i, (smb, conf) in enumerate(product(symbols, configs)):
        conf.update({"batch_id": i // args.batch_size, "symbol": smb})
        tasks.append(conf)

    # dump json
    with open(args.out_file, "w") as f:
        json.dump(tasks, f, sort_keys=True)
