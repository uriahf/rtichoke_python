import plotly.graph_objects as go
from plotly.subplots import make_subplots

from rtichoke.rtichoke_curves.plotly_helper_functions import (
    create_non_interactive_curve,
    create_interactive_marker,
    create_reference_lines_for_plotly
)

def create_plotly_curve(rtichoke_curve_dict):
    """

    Parameters
    ----------
    rtichoke_curve_dict :
        

    Returns
    -------

    """

    # reference_data, 
    # performance_data_ready_for_curve, 
    # group_colors_vec, 
    # axis_ranges

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
                                        visible = False,
                                        args=[None, {"frame": {"duration": 500, "redraw": False}}])])]
                                        }

    for reference_group in list(rtichoke_curve_dict["group_colors_vec"].keys()):
        print("Check if " + reference_group + " Exists")

        if (rtichoke_curve_dict["perf_dat_type"] not in ["several models", "several populations"]) :
            interactive_marker_color = "#f6e3be"
        else :
            interactive_marker_color = rtichoke_curve_dict["group_colors_vec"][reference_group]
        if not (rtichoke_curve_dict["reference_data"].empty) :
            if any(rtichoke_curve_dict["reference_data"]["reference_group"] == reference_group):
                print(reference_group + " Exists :D")
                reference_data_list.append(
                    create_reference_lines_for_plotly(
                    rtichoke_curve_dict["reference_data"][rtichoke_curve_dict["reference_data"]["reference_group"] == reference_group], 
                    rtichoke_curve_dict["group_colors_vec"][reference_group]
                ))
        if any(rtichoke_curve_dict["performance_data_ready_for_curve"]["reference_group"] == reference_group):
            non_interactive_curve.append(
                create_non_interactive_curve(
                    rtichoke_curve_dict["performance_data_ready_for_curve"][rtichoke_curve_dict["performance_data_ready_for_curve"]["reference_group"] == reference_group], 
                    rtichoke_curve_dict["group_colors_vec"][reference_group]
                )
            )
        if any(rtichoke_curve_dict["performance_data_ready_for_curve"]["reference_group"] == reference_group):
            interactive_marker.append(
                create_interactive_marker(
                    rtichoke_curve_dict["performance_data_ready_for_curve"][rtichoke_curve_dict["performance_data_ready_for_curve"]["reference_group"] == reference_group], 
                    interactive_marker_color,
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
            'prefix': rtichoke_curve_dict["animation_slider_prefix"],
            'visible': True,
            'xanchor': 'left'
        },
        'transition': {'duration': 300, 'easing': 'linear'},
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.1,
        'y': 0,
        'steps': []
    }

    for k in range(len(
    rtichoke_curve_dict["performance_data_ready_for_curve"]["stratified_by"].unique()
    )):
        frame_data = reference_data_list + non_interactive_curve
        for reference_group in list(rtichoke_curve_dict["group_colors_vec"].keys()):
            if (rtichoke_curve_dict["perf_dat_type"] not in ["several models", "several populations"]) :
                interactive_marker_color = "#f6e3be"
            else :
                interactive_marker_color = rtichoke_curve_dict["group_colors_vec"][reference_group]

            if any(rtichoke_curve_dict["performance_data_ready_for_curve"]["reference_group"] == reference_group):
                frame_data.append(
                    create_interactive_marker(
                        rtichoke_curve_dict["performance_data_ready_for_curve"][rtichoke_curve_dict["performance_data_ready_for_curve"]["reference_group"] == reference_group], 
                        interactive_marker_color,
                        k)
                    )
        frames.append(go.Frame(data=frame_data, name = str(k)))
        slider_step = {'args': [
            [k],
            {'frame': {'duration': 300, 'redraw': False},
            'mode': 'immediate'}
        ],
        'label': str(rtichoke_curve_dict["performance_data_ready_for_curve"]["stratified_by"].unique()[k]),
        'method': 'animate'}
        sliders_dict['steps'].append(slider_step)                                        

    curve_layout["sliders"] = [sliders_dict]
    fig = go.Figure(data = reference_data_list + non_interactive_curve + interactive_marker, 
                layout = curve_layout,
                frames=frames)

    plotly_curve = fig
    fig.update_xaxes(zeroline=True, range = rtichoke_curve_dict["axes_ranges"]['xaxis'],
    zerolinewidth=1, zerolinecolor='black', fixedrange=True)
    fig.update_yaxes(zeroline=True, range = rtichoke_curve_dict["axes_ranges"]['yaxis'], 
    zerolinewidth=1, zerolinecolor='black', fixedrange=True)
    return plotly_curve





def plot_decision_combined_curve(rtichoke_decision_combined_curve_dict):
    """

    Parameters
    ----------
    rtichoke_decision_combined_curve_dict :
        

    Returns
    -------

    """

    decision_curve_combined = make_subplots(
        rows=2, cols=1, 
        subplot_titles = ('Interventions Avoided (per 100)', 'Net Benefit'),
        shared_xaxes = True
        )

    reference_conventional_data_list = []
    non_interactive_conventional_curve = []
    non_interactive_interventions_avoided_curve = []
    interactive_conventional_marker = []
    interactive_interventions_avoided_marker = []

    curve_layout_subplot = {
    "xaxis": {"showgrid": False,
                },
    "yaxis": {"showgrid": False},
    "plot_bgcolor": "rgba(0, 0, 0, 0)", 
    "showlegend": True,
    "updatemenus": [dict(type="buttons",
                          buttons=[dict(label="Play",
                                        method="animate",
                                        visible = False,
                                        args=[None, {"frame": {"duration": 500, "redraw": False}}])])]
                                        }

    # decision_curve_combined['layout'] = curve_layout_subplot                                        

    for reference_group in list(rtichoke_decision_combined_curve_dict["group_colors_vec"].keys()):
            print("Check if " + reference_group + " Exists")

            if (rtichoke_decision_combined_curve_dict["perf_dat_type"] not in ["several models", "several populations"]) :
                interactive_marker_color = "#f6e3be"
            else :
                interactive_marker_color = rtichoke_decision_combined_curve_dict["group_colors_vec"][reference_group]
            if not (rtichoke_decision_combined_curve_dict["reference_data"]["conventional"].empty) :
                if any(rtichoke_decision_combined_curve_dict["reference_data"]["conventional"]["reference_group"] == reference_group):
                    print(reference_group + " Exists :D")
                    # reference_conventional_data_list.append(
                    #     create_reference_lines_for_plotly(
                    #     rtichoke_decision_combined_curve_dict["reference_data"]["conventional"][rtichoke_decision_combined_curve_dict["reference_data"]["conventional"]["reference_group"] == reference_group], 
                    #     rtichoke_decision_combined_curve_dict["group_colors_vec"][reference_group]
                    # ))
                    decision_curve_combined.add_trace(create_reference_lines_for_plotly(
                        rtichoke_decision_combined_curve_dict["reference_data"]["conventional"][rtichoke_decision_combined_curve_dict["reference_data"]["conventional"]["reference_group"] == reference_group], 
                        rtichoke_decision_combined_curve_dict["group_colors_vec"][reference_group]
                    ), row=2, col=1)
            if any(rtichoke_decision_combined_curve_dict["performance_data_ready_for_curve"]["conventional"]["reference_group"] == reference_group):
                # non_interactive_conventional_curve.append(
                #     create_non_interactive_curve(
                #         rtichoke_decision_combined_curve_dict["performance_data_ready_for_curve"]["conventional"][rtichoke_decision_combined_curve_dict["performance_data_ready_for_curve"]["conventional"]["reference_group"] == reference_group], 
                #         rtichoke_decision_combined_curve_dict["group_colors_vec"][reference_group]
                #     )
                # )
                decision_curve_combined.add_trace(create_non_interactive_curve(
                        rtichoke_decision_combined_curve_dict["performance_data_ready_for_curve"]["conventional"][rtichoke_decision_combined_curve_dict["performance_data_ready_for_curve"]["conventional"]["reference_group"] == reference_group], 
                        rtichoke_decision_combined_curve_dict["group_colors_vec"][reference_group]
                    ), row=2, col=1)
                decision_curve_combined.add_trace(create_interactive_marker(
                        rtichoke_decision_combined_curve_dict["performance_data_ready_for_curve"]["conventional"][rtichoke_decision_combined_curve_dict["performance_data_ready_for_curve"]["conventional"]["reference_group"] == reference_group], 
                        interactive_marker_color,
                        0), row=2, col=1
                    )
            if any(rtichoke_decision_combined_curve_dict["performance_data_ready_for_curve"]["interventions avoided"]["reference_group"] == reference_group):
                decision_curve_combined.add_trace(create_non_interactive_curve(
                        rtichoke_decision_combined_curve_dict["performance_data_ready_for_curve"]["interventions avoided"][rtichoke_decision_combined_curve_dict["performance_data_ready_for_curve"]["interventions avoided"]["reference_group"] == reference_group], 
                        rtichoke_decision_combined_curve_dict["group_colors_vec"][reference_group]
                    ), row=1, col=1)
                decision_curve_combined.add_trace(create_interactive_marker(
                        rtichoke_decision_combined_curve_dict["performance_data_ready_for_curve"]["interventions avoided"][rtichoke_decision_combined_curve_dict["performance_data_ready_for_curve"]["interventions avoided"]["reference_group"] == reference_group], 
                        interactive_marker_color,
                        0), row=1, col=1
                    )

    frames = []        

    sliders_dict = {
        # 'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 20},
            'prefix': rtichoke_decision_combined_curve_dict["animation_slider_prefix"],
            'visible': True,
            'xanchor': 'left'
        },
        'transition': {'duration': 300, 'easing': 'linear'},
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.1,
        'y': 0,
        'steps': []
    }        

    for k in range(len(
    rtichoke_decision_combined_curve_dict["performance_data_ready_for_curve"]["conventional"]["stratified_by"].unique()
    )):
        frame_data = decision_curve_combined.to_dict()["data"]
        for reference_group in list(rtichoke_decision_combined_curve_dict["group_colors_vec"].keys()):
            if (rtichoke_decision_combined_curve_dict["perf_dat_type"] not in ["several models", "several populations"]) :
                interactive_marker_color = "#f6e3be"
            else :
                interactive_marker_color = rtichoke_decision_combined_curve_dict["group_colors_vec"][reference_group]

            if any(rtichoke_decision_combined_curve_dict["performance_data_ready_for_curve"]["conventional"]["reference_group"] == reference_group):
                frame_data.append(
                    create_interactive_marker(
                        rtichoke_decision_combined_curve_dict["performance_data_ready_for_curve"]["conventional"][rtichoke_decision_combined_curve_dict["performance_data_ready_for_curve"]["conventional"]["reference_group"] == reference_group], 
                        interactive_marker_color,
                        k)
                    )
        frames.append(go.Frame(data=frame_data, name = str(k)))
        slider_step = {'args': [
            [k],
            {'frame': {'duration': 300, 'redraw': False},
            'mode': 'immediate'}
        ],
        'label': str(rtichoke_decision_combined_curve_dict["performance_data_ready_for_curve"]["conventional"]["stratified_by"].unique()[k]),
        'method': 'animate'}
        sliders_dict['steps'].append(slider_step)

    # curve_layout_subplot["sliders"] = [sliders_dict]
    
    fig = go.Figure(data = reference_conventional_data_list + non_interactive_conventional_curve, 
                # layout = curve_layout_subplot,
                frames=frames)    

    decision_curve_combined.update(frames=frames)

    decision_curve_combined['layout'].update(
        # updatemenus=updatemenus,
        sliders=sliders_dict)


    plotly_curve = fig
    fig.update_xaxes(zeroline=True, range = rtichoke_decision_combined_curve_dict["axes_ranges"]["conventional"]['xaxis'],
    zerolinewidth=1, zerolinecolor='black', fixedrange=True)
    fig.update_yaxes(zeroline=True, range = rtichoke_decision_combined_curve_dict["axes_ranges"]["conventional"]['yaxis'], 
    zerolinewidth=1, zerolinecolor='black', fixedrange=True)
    

    return decision_curve_combined


# TODO Slider with subplots