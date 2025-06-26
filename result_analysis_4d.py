import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly
import seaborn as sns
from itertools import product
# from pygmo import non_dominated_front_3d

from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
import numpy as np


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


# Path to the .pkl file
res = 'results_dbscan.pkl'

# Load the dictionary from the .pkl file
with open(res, 'rb') as file:
    results = pickle.load(file)

# ADD SOME EXTRA MAN
fin_res = {}
for uid, vals in results.items():
    # vals['comb'] = (vals['clusT_total']/vals['Real'])
    fin_res[uid] = vals

res = fin_res

# Path to the .pkl file
ref = 'reference_dbscan.pkl'

# Load the dictionary from the .pkl file
with open(ref, 'rb') as file:
    reference = pickle.load(file)

# Assuming all results DataFrames have same columns:
sample_result = next(iter(results.values()))
candidate_metrics = sample_result.columns.tolist()

exclude = ['IOU_GT', 'IOU_PRED', 'eval', 'ouratio', 'Miss', 'over', 'under', 'overde', 'VI', 'TP', 'FP', 'FN',
           'Precision', 'Recall', 'F1',
           'StopsH', 'RecordsH', 'UqH']  # Customize!
candidate_metrics = [m for m in candidate_metrics if m not in exclude]
print("Candidate Metrics:", candidate_metrics)

combo_scores = {}
precision_scores = {}
recall_scores = {}
hypervolumes = {}
distances_to_best = {}
best_pareto_type = {}
best_lift = {}
best_percentile_rank = {}
best_hit_score = {}
avg_pareto_sizes = {}

sort_order = [False, False]
compare_func = lambda y, best_y: y > best_y
SET_SIZE = 4
for xval, yval, zval, wval in combinations(candidate_metrics, SET_SIZE):
    if len(set([xval, yval, zval, wval])) < SET_SIZE:
        continue

    best_variant_score = -np.inf
    best_variant_data = {}

    top3_scores = []
    precisions = []
    recalls = []
    hypervolume_vals = []
    distance_vals = []
    pareto_sizes = []
    lifts = []
    percentile_ranks = []
    hit_scores = []

    for person in reference.index:

        x_values = [x.loc[person][xval] for x in results.values()]
        y_values = [x.loc[person][yval] for x in results.values()]
        if SET_SIZE >= 3:
            z_values = [x.loc[person][zval] for x in results.values()]
            if SET_SIZE >= 4:
                w_values = [x.loc[person][wval] for x in results.values()]

        iou_scores = [(2 * x.loc[person]['IOU_GT'] * x.loc[person]['IOU_PRED']) /
                      (x.loc[person]['IOU_GT'] + x.loc[person]['IOU_PRED'] + 1e-6) for x in results.values()]
        annot = list(results.keys())

        # correlation = abs(np.corrcoef(x_values, y_values)[0, 1])
        # if correlation > 0.95:
        #     print(xval,yval,correlation,"OUT")
        #     continue

        data = pd.DataFrame({
            'x': x_values,
            'y': y_values,
            'z': z_values,
            'w': w_values,
            'iou': iou_scores,
            'annot': annot
        })

        # Pareto front extraction
        data_sorted = data.reset_index(drop=True)
        points = data_sorted[['x', 'y', 'z', 'w']].values
        is_pareto = non_dominated_front_3d(points)
        data_sorted['pareto'] = is_pareto

        pareto_df = data_sorted[data_sorted['pareto']]
        non_pareto_df = data_sorted[~data_sorted['pareto']]
        if pareto_df.empty:
            continue

        pareto_df = pareto_df.sort_values('iou', ascending=False).reset_index(drop=True)
        top3 = pareto_df.head(3)
        top3_scores.append(top3['iou'].mean())

        # Precision and recall
        N_prec = 1
        N_recall = 10
        topN_idx = data.sort_values('iou', ascending=False).head(N_recall).index
        topN_idx_prec = data.sort_values('iou', ascending=False).head(N_prec).index
        pareto_idx = data_sorted[data_sorted['pareto']].index
        topN_in_pareto = [i for i in topN_idx if i in pareto_idx]  # how many max are in pareto
        topN_in_pareto_prec = [i for i in topN_idx_prec if i in pareto_idx]  # how many max are in pareto
        precision = len(topN_in_pareto_prec) / N_prec  # if max is in pareto
        recall = 1 if len(topN_in_pareto) > 0 else 0  # if at least one of top 10 is in max
        precisions.append(precision)
        recalls.append(recall)

        # Normalize hypervolume using bounding box
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)

        pareto_sorted = pareto_df.sort_values('x', ascending=True)
        hv = 0
        x_prev = x_min
        for _, row in pareto_sorted.iterrows():
            width = row['x'] - x_prev
            height = y_max - row['y']
            hv += width * height
            x_prev = row['x']

        # Normalize to [0, 1] using total area of bounding box
        area = (x_max - x_min) * (y_max - y_min)
        hv_norm = hv / area if area > 0 else 0
        hypervolume_vals.append(hv_norm)

        # Best IOUs
        best_total_iou = data['iou'].max()
        best_pareto_iou = pareto_df['iou'].max()
        dist = abs(best_total_iou - best_pareto_iou)
        distance_vals.append(dist)

        # Score lift over basline
        numeric_cols = pareto_df.select_dtypes(include='number').columns
        lift = pareto_df['iou'].mean() - non_pareto_df['iou'].mean()
        lifts.append(lift)

        # Percentiles
        organised = data_sorted.sort_values('iou', ascending=False)
        organised['num'] = range(organised.shape[0])
        topsis = (organised[organised['pareto']].sort_values('num')['num'].iloc[0] + 1) / organised.shape[0]
        percentile_ranks.append(topsis)

        pareto_sizes.append(pareto_df.shape[0] / data_sorted.shape[0])
        scaler = MinMaxScaler()
        scaler.fit_transform(data_sorted.sort_values('iou')[['x', 'y', 'z', 'w', 'iou', 'pareto']])

    combo_scores[(xval, yval, zval, wval)] = str((np.mean(top3_scores), np.std(top3_scores)))
    precision_scores[(xval, yval, zval, wval)] = np.mean(precisions)
    recall_scores[(xval, yval, zval, wval)] = np.mean(recalls)
    hypervolumes[(xval, yval, zval, wval)] = np.mean(hypervolume_vals)
    distances_to_best[(xval, yval, zval, wval)] = np.mean(distance_vals)
    best_lift[(xval, yval, zval, wval)] = np.mean(lifts)
    best_percentile_rank[(xval, yval, zval, wval)] = np.mean(percentile_ranks)
    avg_pareto_sizes[(xval, yval, zval, wval)] = np.mean(pareto_sizes)

    print(f"{xval}, {yval}, {zval}, {wval} | Top-3 IOU: {combo_scores[(xval, yval, zval, wval)]}, "
          f"P@{N_prec}: {precision_scores[(xval, yval, zval, wval)]:.2f}, R@{N_recall}: {recall_scores[(xval, yval, zval, wval)]:.2f}, "
          f"HV: {hypervolumes[(xval, yval, zval, wval)]:.2f}, Dist: {distances_to_best[(xval, yval, zval, wval)]:.2f},"
          f"Lift: {best_lift[(xval, yval, zval, wval)]:.2f}, Percentile: {best_percentile_rank[(xval, yval, zval, wval)]:.2f}, 'Size:': {avg_pareto_sizes[(xval, yval, zval, wval)]:.2f}")
combined = pd.concat([pd.DataFrame().from_dict(x, orient='index') for x in
                      [combo_scores, precision_scores, recall_scores, hypervolumes, distances_to_best, best_lift,
                       best_percentile_rank]], axis=1)
combined.columns = ['TOP3', 'P', 'R', 'H', 'D', 'L', 'Perc']
combined.sort_values('TOP3')
