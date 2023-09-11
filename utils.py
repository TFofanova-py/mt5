from datetime import datetime
import logging
from time import sleep


def wait_for_next_hour(verbose=False) -> None:
    now = datetime.now()
    seconds_to_next_hour = 60 * 60 - now.minute * 60 - now.second + 1

    msg = f"waiting for {seconds_to_next_hour // 60} minutes for the next hour"
    logging.info(msg)
    if verbose:
        print(msg)

    sleep(seconds_to_next_hour)
