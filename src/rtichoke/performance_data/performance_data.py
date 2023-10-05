"""
A module for Performance Data
"""

from typing import Dict, List
from pandas.core.frame import DataFrame
import pandas as pd
from rtichoke.helpers.send_post_request_to_r_rtichoke import send_requests_to_rtichoke_r


def prepare_performance_data(
    probs: Dict[str, List[float]],
    reals: Dict[str, List[int]],
    stratified_by: str = "probability_threshold",
    url_api: str = "http://localhost:4242/",
) -> DataFrame:
    """Prepare Performance Data

    Args:
        probs (Dict[str, List[float]]): _description_
        reals (Dict[str, List[int]]): _description_
        stratified_by (str, optional): _description_. Defaults to "probability_threshold".
        url_api (_type_, optional): _description_. Defaults to "http://localhost:4242/".

    Returns:
        DataFrame: _description_
    """
    rtichoke_response = send_requests_to_rtichoke_r(
        dictionary_to_send={
            "probs": probs,
            "reals": reals,
            "stratified_by": stratified_by,
        },
        url_api=url_api,
        endpoint="prepare_performance_data",
    )

    performance_data = pd.DataFrame(
        rtichoke_response.json(), columns=list(rtichoke_response.json()[0].keys())
    )
    return performance_data
