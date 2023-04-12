"""
A module for ROC Curves
"""

from typing import Dict, List, Optional
from plotly.graph_objs._figure import Figure
from pandas.core.frame import DataFrame
from rtichoke.helpers.send_post_request_to_r_rtichoke import create_rtichoke_curve
from rtichoke.helpers.send_post_request_to_r_rtichoke import plot_rtichoke_curve


def create_roc_curve(
    probs: Dict[str, List[float]],
    reals: Dict[str, List[int]],
    by: float = 0.01,
    stratified_by: str = "probability_threshold",
    size: Optional[int] = None,
    color_values: List[str] = None,
    url_api: str = "http://localhost:4242/",
) -> Figure:
    """Create ROC Curve

    Args:
        probs (Dict[str, List[float]]): _description_
        reals (Dict[str, List[int]]): _description_
        by (float, optional): _description_. Defaults to 0.01.
        stratified_by (str, optional): _description_. Defaults to "probability_threshold".
        size (Optional[int], optional): _description_. Defaults to None.
        color_values (List[str], optional): _description_. Defaults to None.
        url_api (_type_, optional): _description_. Defaults to "http://localhost:4242/".

    Returns:
        Figure: _description_
    """
    fig = create_rtichoke_curve(
        probs,
        reals,
        by=by,
        stratified_by=stratified_by,
        size=size,
        color_values=color_values,
        url_api=url_api,
        curve="roc",
    )
    return fig


def plot_roc_curve(
    performance_data: DataFrame,
    size: Optional[int] = None,
    color_values: List[str] = None,
    url_api: str = "http://localhost:4242/",
) -> Figure:
    """Plot ROC Curve

    Args:
        performance_data (DataFrame): _description_
        size (Optional[int], optional): _description_. Defaults to None.
        color_values (List[str], optional): _description_. Defaults to None.
        url_api (_type_, optional): _description_. Defaults to "http://localhost:4242/".

    Returns:
        Figure: _description_
    """
    fig = plot_rtichoke_curve(
        performance_data,
        size=size,
        color_values=color_values,
        url_api=url_api,
        curve="roc",
    )
    return fig
