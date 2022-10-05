from rtichoke.rtichoke_curves.plotly_helper_functions import (
    create_non_interactive_curve,
    create_interactive_marker,
    create_plotly_curve
)

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
