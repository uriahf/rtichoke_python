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


def create_plot_dict(curve_type, stratification):
    plot_dicts = {
        "ROC": {
            "x": "FPR",
            "y": "Sensitivity",
            "reference": {"x": [0, 1], "y": [0, 1]},
            "xlabel": "1-Specificity",
            "ylabel": "Sensitivity",
            "legend": "bottom_right",
            "hover_info": [
                ("Dataset", "@Population"),
                ("probability_threshold", "@probability_threshold{0.000}"),
                ("PPCR", "@ppcr{0.000}"),
                ("Predicted positive", "@predicted_positives"),
                ("FPR", "@FPR{0.000}"),
                ("Sensitivity", "@Sensitivity{0.000}"),
            ],
        },
        "LIFT": {
            "x": "ppcr",
            "y": "lift",
            "reference": {"x": [0, 1], "y": [1, 1]},
            "xlabel": "ppcr",
            "ylabel": "lift",
            "legend": "top_right",
            "hover_info": [
                ("Dataset", "@Population"),
                ("probability_threshold", "@probability_threshold{0.000}"),
                ("PPCR", "@ppcr{0.000}"),
                ("Predicted positive", "@predicted_positives"),
                ("lift", "@lift{0.000}"),
            ],
        },
        "PR": {
            "x": "PPV",
            "y": "Sensitivity",
            "reference": {"x": [0, 0], "y": [0, 0]},
            "xlabel": "Precision",
            "ylabel": "Recall",
            "legend": "top_right",
            "hover_info": [
                ("Dataset", "@Population"),
                ("probability_threshold", "@probability_threshold{0.000}"),
                ("PPCR", "@ppcr{0.000}"),
                ("Predicted positive", "@predicted_positives"),
                ("Precision", "@PPV{0.000}"),
                ("Recall", "@Sensitivity{0.000}"),
            ],
        },
    }

    return (
        plot_dicts[curve_type]["x"],
        plot_dicts[curve_type]["y"],
        plot_dicts[curve_type],
    )


def select_data_table(self, x, y, stratification="probability_threshold"):
    df = (
        self.performance_table_pt
        if stratification == "probability_threshold"
        else self.performance_table_ppcr
    )
    cols = list(
        set(
            ["Population", "predicted_positives", "probability_threshold", "ppcr", x, y]
        )
    )
    return df[cols]


def create_pops_and_colors(df):
    pops = df.Population.unique()
    colors = palette[0 : len(pops)]
    color_map = CategoricalColorMapper(factors=pops, palette=palette)

    return pops, colors, color_map
