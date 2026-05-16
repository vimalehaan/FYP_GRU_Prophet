import numpy as np

def create_residual_sequences(
    df,
    features,
    target,
    input_window=96,
    forecast_horizon=96
):

    X, y = [], []
    sample_cids = []

    for cid, group in df.groupby("container_id"):

        group = group.sort_values("time_stamp")

        values = group[features].values
        target_values = group[target].values

        max_idx = (
            len(group)
            - input_window
            - forecast_horizon
        )

        for i in range(max_idx):

            X.append(
                values[
                    i : i + input_window
                ]
            )

            y.append(
                target_values[
                    i + input_window :
                    i + input_window + forecast_horizon
                ]
            )

            sample_cids.append(cid)

    return (
        np.array(X),
        np.array(y),
        np.array(sample_cids)
    )