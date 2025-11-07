import numpy as np
import pandas as pd


def non_dominated_front_3d(points: np.ndarray) -> np.ndarray:
    """
    Boolean mask of 3-D Pareto-efficient points (maximise all axes).

    Parameters
    ----------
    points : (N, 3) array_like
        Rows are candidate points; larger values are better.

    Returns
    -------
    is_efficient : (N,) bool ndarray
        True for Pareto-optimal rows.
    """
    points = np.asarray(points)
    N = points.shape[0]
    is_efficient = np.ones(N, dtype=bool)

    for i, c in enumerate(points):
        if not is_efficient[i]:
            continue
        # a point y is dominated by c  ⇔  c ≥ y  &  c > y in ≥1 dim
        dominated = (points <= c).all(axis=1) & (points < c).any(axis=1)
        is_efficient[dominated] = False

    return is_efficient


def pareto_mask(points):
    is_efficient = np.ones(points.shape[0], dtype=bool)
    for i, c in enumerate(points):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(points[is_efficient] > c, axis=1)
            is_efficient[i] = True
    return is_efficient


def global_sensitivity_analysis(results, metric='Stops',
                                smooth_window=3,
                                sens_quantile=0.25,
                                var_quantile=0.25
                                ):
    """
    Perform global sensitivity analysis for stop detection results.
    Identifies parameter regions where the number of detected stops is stable.

    Args:
        results (dict): {parameter_tuple_str: DataFrame of results per user}
        param_names (tuple): names of parameters in each key (parsed from string)
        metric (str): metric to analyze, e.g. 'Stops'
        stability_threshold (float): threshold for relative change marking "stable region"
    """

    # Parse parameters and collect global metric
    param_list, metric_values = [], []
    for k, df in results.items():
        try:
            # Convert key string "[50, 3, 50, '3T']" → (50, 3, 50, '3T')
            params = eval(k)

            # Aggregate metric per configuration
            mean_metric = df[metric].mean()
            param_list.append(params)
            metric_values.append(mean_metric)
        except Exception as e:
            print(f"Skipping {k}: {e}")

    param_df = pd.DataFrame(param_list)
    param_df[metric] = metric_values
    param_cols = [x for x in param_df.columns if type(x) == int]

    picked_params = []
    # ---------- Per-parameter analysis ----------
    for p in param_cols:
        if param_df[p].nunique() <= 2:
            pick = param_df[p].unique().min()
            picked_params.append(pick)
            continue

        # Sort by this parameter
        tmp = pd.DataFrame(param_df.groupby(p)[metric].mean().sort_index())

        # Compute absolute diffs
        tmp['diff_prev'] = tmp[metric].diff().abs()  # |x_i - x_{i-1}|
        tmp['diff_next'] = tmp[metric].shift(-1).sub(tmp[metric]).abs()  # |x_{i+1} - x_i|

        # Stability score = local variability
        tmp['stability'] = tmp[['diff_prev', 'diff_next']].sum(axis=1)

        # Normalize (0–1) to make them comparable
        tmp['stability_norm'] = (tmp['stability'] - tmp['stability'].min()) / (
                    tmp['stability'].max() - tmp['stability'].min())
        tmp['Stops_norm'] = (tmp['Stops'] - tmp['Stops'].min()) / (tmp['Stops'].max() - tmp['Stops'].min())

        # Combined score
        tmp['score'] = tmp['Stops_norm'] - tmp['stability_norm']

        # Select best compromise
        pick = tmp['score'].idxmax()
        if pd.isna(pick):
            pick = param_df[p].unique().min()
        picked_params.append(pick)

    return str(picked_params)
