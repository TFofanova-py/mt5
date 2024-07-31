from datetime import datetime
import logging
from time import sleep
from typing import Union
from pair.multiindpair import BasePair, MultiIndPair, RelatedPair


def wait_for_next_hour(verbose=False) -> None:
    now = datetime.now()
    seconds_to_next_hour = 60 * 60 - now.minute * 60 - now.second + 1

    msg = f"waiting for {seconds_to_next_hour // 60} minutes for the next hour"
    logging.info(msg)
    if verbose:
        print(msg)

    sleep(seconds_to_next_hour)


def sleep_with_dummy_requests(delay: int, p: Union[BasePair, MultiIndPair, RelatedPair], **kwargs):
    if kwargs.get("capital_conn") is None:
        sleep(delay * 60)

    else:
        request_freq: int = 9
        for _ in range(delay // request_freq):
            sleep(request_freq * 60)
            p.get_historical_data(**kwargs)  # dummy request to don't lose the api session

        sleep((delay % request_freq) * 60)
