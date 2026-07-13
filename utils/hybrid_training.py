"""Prophet residual generation for Hybrid GRU training."""

from __future__ import annotations

import pandas as pd
from prophet import Prophet


def generate_prophet_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit per-container Prophet models on the train period and extract residuals.

    Parameters
    ----------
    df:
        Train-period dataframe with ``container_id``, ``time_stamp``,
        and ``cpu_scaled`` columns.

    Returns
    -------
    DataFrame with ``prophet_pred`` and ``residual`` columns appended.
    """
    prophet_outputs = []

    for _, group in df.groupby("container_id"):
        group = group.sort_values("time_stamp")

        prophet_df = pd.DataFrame({
            "ds": group["time_stamp"],
            "y": group["cpu_scaled"],
        })

        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=False,
        )
        model.fit(prophet_df)

        forecast = model.predict(prophet_df[["ds"]])

        group["prophet_pred"] = forecast["yhat"].values
        group["residual"] = (
            group["cpu_scaled"]
            - group["prophet_pred"]
        )

        prophet_outputs.append(group)

    return pd.concat(
        prophet_outputs,
        ignore_index=True,
    )
