import requests
import pandas as pd
from rtichoke.helpers.exported_functions import create_plotly_curve
from rtichoke.helpers.send_post_request_to_r_rtichoke import create_rtichoke_curve
from rtichoke.helpers.send_post_request_to_r_rtichoke import plot_rtichoke_curve
from pandas.core.frame import DataFrame
from plotly.graph_objs._figure import Figure
from typing import Dict, List, Optional


def create_decision_curve(probs: Dict[str, List[float]], reals: Dict[str, List[int]], type: str="conventional",
    min_p_threshold: float = 0,
    max_p_threshold: float = 1, by: float = 0.01, stratified_by: str = "probability_threshold", size: Optional[int]= None, color_values: List[str] = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#07004D", "#E6AB02",
                "#FE5F55", "#54494B", "#006E90", "#BC96E6", "#52050A", "#1F271B", "#BE7C4D",
                "#63768D", "#08A045", "#320A28", "#82FF9E", "#2176FF", "#D1603D", "#585123"] ,url_api: str = "http://localhost:4242/") -> Figure:
        
        if type == "conventional":
            curve = "decision"
        else:
            curve = "interventions avoided"
        
        fig = create_rtichoke_curve(
            probs, 
            reals, 
            by = by,
            stratified_by = stratified_by,
            size = size,
            color_values = color_values,
            url_api = url_api,
            curve = curve, 
            min_p_threshold = min_p_threshold,
            max_p_threshold = max_p_threshold)
        return fig



def plot_decision_curve(performance_data: DataFrame, type: str,
    min_p_threshold: int = 0,
    max_p_threshold: int = 1, size: Optional[int]= None, color_values: List[str] = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#07004D", "#E6AB02",
                "#FE5F55", "#54494B", "#006E90", "#BC96E6", "#52050A", "#1F271B", "#BE7C4D",
                "#63768D", "#08A045", "#320A28", "#82FF9E", "#2176FF", "#D1603D", "#585123"], url_api: str = "http://localhost:4242/") -> Figure:
    
    if type == "conventional":
        curve = "decision"
    else:
        curve = "interventions avoided"
    
    fig = plot_rtichoke_curve(
            performance_data, 
            size = size,
            color_values = color_values,
            url_api = url_api,
            curve = curve,
            min_p_threshold = min_p_threshold,
            max_p_threshold = max_p_threshold)
    return fig