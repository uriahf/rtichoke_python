from datetime import datetime


def tprint(string):
    now = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
    print(now + " - " + string)


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
