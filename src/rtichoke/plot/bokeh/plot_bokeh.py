from bokeh.layouts import column
from bokeh.models import (
    ColumnDataSource,
    Slider,
    CustomJS,
    HoverTool,
)
from bokeh.plotting import figure, output_file, save
from collections import OrderedDict
from .create_bokeh_plot_dict import *


def plot_bokeh(self, generic_plot_dict, filename=None):
    curve_type = generic_plot_dict["curve_type"]
    stratification = generic_plot_dict["stratification"]

    graph_meta = create_bokeh_plot_dict(generic_plot_dict)
    x = graph_meta["x"]
    y = graph_meta["y"]
    df = self.select_data_table(x=x, y=y, stratification=stratification)
    pops, colors, color_map = create_pops_and_colors(df)

    lines_source = ColumnDataSource(
        OrderedDict(
            Population=[df[df["Population"] == pop]["Population"] for pop in pops],
            xs=[df[df["Population"] == pop][x] for pop in pops],
            ys=[df[df["Population"] == pop][y] for pop in pops],
            colors=colors,
            legend_labels=pops,
        )
    )

    # Create plots and widgets
    if filename:
        output_file(filename)

    plot = figure(title=f"{curve_type} curve", tools="box_zoom")
    plot.multi_line(
        "xs",
        "ys",
        source=lines_source,
        line_width=3,
        line_alpha=0.5,
        line_color="colors",
        legend_field="legend_labels",
    )
    plot.legend.location = graph_meta["legend"]
    plot.line(
        graph_meta["reference"]["x"],
        graph_meta["reference"]["y"],
        line_width=1,
        color="gray",
    )  # reference line

    # Create Slider object
    slider = Slider(
        start=0,
        end=1,
        value=df.dropna()[stratification].min(),
        step=self.by,
        title=stratification,
    )

    # add scatter
    filtered_scatter = [
        True if th == slider.value else False for th in df[stratification]
    ]
    scatter_source = ColumnDataSource(data=df.loc[filtered_scatter])
    scatter = plot.scatter(
        x=x,
        y=y,
        source=scatter_source,
        size=11,
        fill_alpha=0.85,
        line_color="black",
        color={"field": "Population", "transform": color_map},
    )

    # add hover data
    hover = HoverTool(
        renderers=[scatter],
        tooltips=graph_meta["hover_info"],
    )
    plot.add_tools(hover)

    # Adding callback code
    source = ColumnDataSource(data=df.set_index("Population"))
    callback = CustomJS(
        args=dict(source=source, scatter_source=scatter_source, val=slider),
        code=create_JS_code(x=x, y=y, stratification=stratification),
    )

    slider.js_on_change("value", callback)

    # customize plots and finish
    plot.xaxis.axis_label = graph_meta["xlabel"]
    plot.yaxis.axis_label = graph_meta["ylabel"]
    layout = column(slider, plot)

    if filename:
        save(layout)
    else:
        return layout
