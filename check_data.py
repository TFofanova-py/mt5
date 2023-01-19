import numpy as np
import pandas as pd
import scipy.stats as sts
import logging


def fix_missing(data: pd.DataFrame, mandatory_cols: list):  # , n_rows: int):
    n_rows = data.shape[0]  # min(n_rows, data.shape[0])
    col_nums = [data.columns.to_list().index(col) for col in mandatory_cols]

    if data.iloc[-n_rows:, col_nums].isna().sum().sum() > 0:
        logging.info(f"Missings at {data.index[-1]}")

        for i_col in col_nums:
            data.iloc[-n_rows:, i_col].fillna(method="ffill", inplace=True)

    return data


def drop_outlier(data: pd.DataFrame, mandatory_cols: list):  # , n_rows: int):
    n_rows = data.shape[0]  # min(n_rows, data.shape[0])

    m = data.iloc[-n_rows:][mandatory_cols].mean(axis=0)  # vector of means
    sigma = data.iloc[-n_rows:][mandatory_cols].std(axis=0)  # vector of stds

    left = sts.norm.ppf(0.005, loc=m, scale=sigma)  # vector of left quantiles
    right = sts.norm.ppf(0.995, loc=m, scale=sigma)  # vector of right quantiles

    for i in range(len(mandatory_cols)):
        col_num = data.columns.to_list().index(mandatory_cols[i])
        data.iloc[-n_rows:, col_num] = data.iloc[-n_rows:, col_num].apply(lambda x: x if left[i] < x < right[i] else np.NaN)

    return data

