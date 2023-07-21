from bokeh.palettes import Spectral11 as palette
from bokeh.models import CategoricalColorMapper


def create_JS_code(x, y, stratification):
    js_code = f"""    
    const source_data = source.data;
    const pop = source_data.Population;
    const x = source_data.{x};
    const y = source_data.{y};
    const probability_threshold = source_data.probability_threshold;
    const predicted_positives = source_data.predicted_positives;
    const ppcr = source_data.ppcr;
    const pt = val.value;

    const j = val.step.toString().split('.')[1].length;
    const filter = source_data.{stratification}.map((num) => num.toFixed(j) == pt.toFixed(j));   
    
    scatter_source.data = {{
        Population: pop.filter((_, i) => filter[i]),
        {x}: x.filter((_, i) => filter[i]),
        {y}: y.filter((_, i) => filter[i]),
        probability_threshold: probability_threshold.filter((_, i) => filter[i]),
        ppcr: ppcr.filter((_, i) => filter[i]),
        predicted_positives: predicted_positives.filter((_, i) => filter[i])
    }};
    
    scatter_source.change.emit();
    """
    return js_code


_legend_positions = {
    "ROC": "bottom_right",
    "LIFT": "top_right",
    "PR": "top_right",
    "NB": "bottom_left",
}
_generic_hover = [
    ("Dataset", "@Population"),
    ("Prob. threshold", "@probability_threshold{0.000}"),
    ("PPCR", "@ppcr{0.000}"),
    ("Predicted positive", "@predicted_positives"),
]


def create_bokeh_plot_dict(bokeh_plot_dict):
    curve_type = bokeh_plot_dict["curve_type"]

    bokeh_plot_dict["legend"] = _legend_positions[curve_type]
    specific_hover_info = [
        (l, "@" + l + "{0.000}") for l in bokeh_plot_dict["hover_info"]
    ]
    bokeh_plot_dict["hover_info"] = _generic_hover + specific_hover_info

    return bokeh_plot_dict


def create_pops_and_colors(df):
    pops = df.Population.unique()
    colors = palette[0 : len(pops)]
    color_map = CategoricalColorMapper(factors=pops, palette=palette)

    return pops, colors, color_map
