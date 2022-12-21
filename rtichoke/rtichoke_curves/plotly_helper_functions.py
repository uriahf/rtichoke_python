import plotly.graph_objects as go

def create_non_interactive_curve(performance_data_ready_for_curve, reference_group_color):
    """

    Parameters
    ----------
    performance_data_ready_for_curve :
        
    reference_group_color :
        

    Returns
    -------

    """
    performance_data_ready_for_curve = performance_data_ready_for_curve.dropna()
    # print("Print y values non interactive")
    # print(performance_data_ready_for_curve['y'].values)
    # print("Done Printing non interactive")
    non_interactive_curve = go.Scatter(
                x=performance_data_ready_for_curve["x"].values.tolist(),
                y=performance_data_ready_for_curve["y"].values.tolist(),
                mode="markers+lines",
                hoverinfo="text",
                hovertext=performance_data_ready_for_curve["text"].values.tolist(),
                name = "model_non_interactive",
                line=dict(width=2, color=reference_group_color))
    return non_interactive_curve

def create_interactive_marker(performance_data_ready_for_curve, interactive_marker_color, k):
    """

    Parameters
    ----------
    performance_data_ready_for_curve :
        
    reference_group_color :
        
    k :
        

    Returns
    -------

    """
    performance_data_ready_for_curve = performance_data_ready_for_curve.assign(column_name=performance_data_ready_for_curve.loc[:, "y"].fillna(-1))
    # print("Print y values")
    # print(performance_data_ready_for_curve['y'].values)
    # print("Done Printing")

    interactive_marker = go.Scatter(
                x=[performance_data_ready_for_curve["x"].values.tolist()[k]],
                y=[performance_data_ready_for_curve["y"].values.tolist()[k]],
                mode="markers",
                hoverinfo="text",
                hovertext=performance_data_ready_for_curve["text"].values.tolist(),
                name = "model_interactive",
                marker=dict(size = 12, color=interactive_marker_color, line=dict(width=2, color = "black")))
    return interactive_marker



def create_reference_lines_for_plotly(reference_data, reference_line_color):
    """Creates a plotly scatter object of the reference lines

    Parameters
    ----------
    reference_data :
        
    reference_line_color :
        

    Returns
    -------

    """
    reference_lines = go.Scatter(x=reference_data["x"].values.tolist(), 
                     y=reference_data["y"].values.tolist(),
                     mode="lines",
                     hoverinfo="text",
                     hovertext=reference_data["text"].values.tolist(),
                     name="reference_line",
                     line=dict(width=2, color=reference_line_color, 
                     dash = "dot"))
    return reference_lines
