import numpy as np
import MetaTrader5 as mt5

N_RANGES = 300
MIN_PRICE_HIST_PERIOD = 100
HIGHEST_FIB = 0.236
CENTRAL_HIGH_FIB = 0.382
CENTRAL_LOW_FIB = 0.618
LOWEST_FIB = 0.764
NUM_ATTEMPTS_FETCH_PRICES = 10
N_DOWN_PERIODS = (4, np.inf)  # (3, 4, np.inf)

MT5_TIMEFRAME = {"1m": mt5.TIMEFRAME_M1, "2m": mt5.TIMEFRAME_M2, "3m": mt5.TIMEFRAME_M3, "5m": mt5.TIMEFRAME_M5,
                 "6m": mt5.TIMEFRAME_M6, "10m": mt5.TIMEFRAME_M10, "15m": mt5.TIMEFRAME_M15,
                 "20m": mt5.TIMEFRAME_M20, "30m": mt5.TIMEFRAME_M30, "1h": mt5.TIMEFRAME_H1, "2h": mt5.TIMEFRAME_H2,
                 "3h": mt5.TIMEFRAME_H3, "4h": mt5.TIMEFRAME_H4, "6h": mt5.TIMEFRAME_H6, "12h": mt5.TIMEFRAME_H12,
                 "1d": mt5.TIMEFRAME_D1, "1wk": mt5.TIMEFRAME_W1
                 }

CAPITAL_TIMEFRAME = {1: "MINUTE", 5: "MINUTE_5", 15: "MINUTE_15",
                     30: "MINUTE_30", 60: "HOUR", 240: "HOUR_4",
                     1440: "DAY", 10080: "WEEK"
                     }
EPSILON = 1e-4
