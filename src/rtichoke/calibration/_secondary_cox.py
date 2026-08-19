"""Secondary Cox calibration smoothing backed by smoothstate."""

from collections.abc import Callable

import numpy as np
import polars as pl
from smoothstate import smooth_state_cox


def calculate_secondary_cox_smooth(
    df_adj: pl.DataFrame,
    horizon: float,
    performance_type: str,
    *,
    aj_risk_at_horizon: Callable[[pl.DataFrame, float], float],
) -> pl.DataFrame:
    """Calculate the secondary-Cox calibration curve with ``smoothstate``.

    The degenerate/error fallback intentionally mirrors rtichoke's previous
    implementation: return a constant Aalen-Johansen risk curve on [0, 1].
    ``performance_type`` remains in the signature for API compatibility with
    the existing private helper.
    """
    del performance_type
    smooth_frames: list[pl.DataFrame] = []

    for key, group_df in df_adj.group_by("reference_group", maintain_order=True):
        group_name = str(key[0])
        probs = group_df["prob"].to_numpy()
        reals = group_df["real"].to_numpy()
        times = group_df["time"].to_numpy()
        events = (reals == 1).astype(int)

        p_clipped = np.clip(probs, 1e-6, 1 - 1e-6)
        transformed = np.log(-np.log(1 - p_clipped))

        if len(np.unique(transformed)) <= 1 or events.sum() == 0:
            y_est = aj_risk_at_horizon(group_df, horizon)
            xout = np.linspace(0, 1, 101)
            smooth_frames.append(
                pl.DataFrame(
                    {
                        "x": xout,
                        "y": np.full(len(xout), y_est),
                        "reference_group": [group_name] * len(xout),
                    }
                )
            )
            continue

        try:
            smoothed = smooth_state_cox(
                probs=probs,
                times=times,
                events=events,
                horizon=horizon,
                penalizer=0.01,
            )
            smooth_frames.append(
                smoothed.with_columns(pl.lit(group_name).alias("reference_group"))
            )
        except Exception:
            y_est = aj_risk_at_horizon(group_df, horizon)
            xout = np.linspace(0, 1, 101)
            smooth_frames.append(
                pl.DataFrame(
                    {
                        "x": xout,
                        "y": np.full(len(xout), y_est),
                        "reference_group": [group_name] * len(xout),
                    }
                )
            )

    if not smooth_frames:
        return pl.DataFrame(
            schema={
                "x": pl.Float64,
                "y": pl.Float64,
                "reference_group": pl.Utf8,
            }
        )

    return pl.concat(smooth_frames)
