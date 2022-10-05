import plotly.graph_objects as go
import pandas as pd

# Try commit

def create_reference_lines_for_plotly(reference_data, reference_line_color):
    """
    Creates a plotly scatter object of the reference lines
    
    Args:
        `reference_data` stands for the reference_data provided
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

def create_non_interactive_curve(performance_data_ready_for_curve, reference_group_color):
    non_interactive_curve = go.Scatter(
                x=performance_data_ready_for_curve["x"].values.tolist(),
                y=performance_data_ready_for_curve["y"].values.tolist(),
                mode="markers+lines",
                hoverinfo="text",
                hovertext=performance_data_ready_for_curve["text"].values.tolist(),
                name = "model_non_interactive",
                line=dict(width=2, color=reference_group_color))
    return non_interactive_curve

def create_interactive_marker(performance_data_ready_for_curve, reference_group_color, k):
    interactive_marker = go.Scatter(
                x=[performance_data_ready_for_curve["x"].values.tolist()[k]],
                y=[performance_data_ready_for_curve["y"].values.tolist()[k]],
                mode="markers",
                hoverinfo="text",
                hovertext=performance_data_ready_for_curve["text"].values.tolist(),
                name = "model_interactive",
                marker=dict(size = 12, color=reference_group_color, line=dict(width=2, color = "black")))
    return interactive_marker

def create_plotly_curve(reference_data, performance_data_ready_for_curve, group_colors_vec, axis_ranges):
    reference_data_list = []
    non_interactive_curve = []
    interactive_marker = []

    curve_layout = {
    "xaxis": {"showgrid": False,
                },
    "yaxis": {"showgrid": False},
    "plot_bgcolor": "rgba(0, 0, 0, 0)", 
    "showlegend": True,
    "updatemenus": [dict(type="buttons",
                          buttons=[dict(label="Play",
                                        method="animate",
                                        args=[None, {"frame": {"duration": 500, "redraw": False}}])])]}

    for reference_group in list(group_colors_vec.keys()):
        print("Check if " + reference_group + " Exists")
        if any(reference_data["reference_group"] == reference_group):
            print(reference_group + " Exists :D")
            reference_data_list.append(
                create_reference_lines_for_plotly(
                reference_data[reference_data["reference_group"] == reference_group], 
                group_colors_vec[reference_group]
            ))
        if any(performance_data_ready_for_curve["reference_group"] == reference_group):
            non_interactive_curve.append(
                create_non_interactive_curve(
                    performance_data_ready_for_curve[performance_data_ready_for_curve["reference_group"] == reference_group], 
                    group_colors_vec[reference_group]
                )
            )
        if any(performance_data_ready_for_curve["reference_group"] == reference_group):
            interactive_marker.append(
                create_interactive_marker(
                    performance_data_ready_for_curve[performance_data_ready_for_curve["reference_group"] == reference_group], 
                    group_colors_vec[reference_group],
                    0
                )
            )

    frames = []

    sliders_dict = {
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 20},
            'prefix': 'Probability Threshold:',
            'visible': True,
            'xanchor': 'left'
        },
        'transition': {'duration': 300, 'easing': 'cubic-in-out'},
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.1,
        'y': 0,
        'steps': []
    }

    for k in range(100):
        frame_data = reference_data_list + non_interactive_curve
        for reference_group in list(group_colors_vec.keys()):
            if any(performance_data_ready_for_curve["reference_group"] == reference_group):
                frame_data.append(
                    create_interactive_marker(
                        performance_data_ready_for_curve[performance_data_ready_for_curve["reference_group"] == reference_group], 
                        group_colors_vec[reference_group],
                        k)
                    )
        frames.append(go.Frame(data=frame_data, name = str(k)))
        slider_step = {'args': [
            [k],
            {'frame': {'duration': 300, 'redraw': False},
            'mode': 'immediate'}
        ],
        'label': str(k),
        'method': 'animate'}
        sliders_dict['steps'].append(slider_step)                                        

    curve_layout["sliders"] = [sliders_dict]
    fig = go.Figure(data = reference_data_list + non_interactive_curve + interactive_marker, 
                layout = curve_layout,
                frames=frames)

    plotly_curve = fig
    fig.update_xaxes(zeroline=True, range = axis_ranges['xaxis'],
    zerolinewidth=1, zerolinecolor='black', fixedrange=True)
    fig.update_yaxes(zeroline=True, range = axis_ranges['yaxis'], 
    zerolinewidth=1, zerolinecolor='black', fixedrange=True)
    return plotly_curve