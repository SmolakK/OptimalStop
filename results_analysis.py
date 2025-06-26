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

# REFERENCE CALCULATIONS
df = pd.read_csv('data/reference.csv').iloc[:, 1:]
df['datetime'] = pd.to_datetime(df['datetime'])
df.columns = ['user_id', 'datetime', 'labels_reference', 'geometry', 'lon', 'lat']
hours_per_person = df.groupby('user_id').apply(lambda x: (x.datetime.max() - x.datetime.min()).total_seconds()/3600)

# Path to the .pkl file
res = 'results_infostop.pkl'

# Load the dictionary from the .pkl file
with open(res, 'rb') as file:
    results = pickle.load(file)

#ADD SOME EXTRA MAN
fin_res = {}
for uid,vals in results.items():
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

exclude = ['IOU_GT', 'IOU_PRED', 'eval', 'ouratio', 'Miss', 'over', 'under', 'overde', 'VI', 'TP','FP','FN','Precision','Recall','F1',
           'StopsH','RecordsH','UqH','clus_points','clus_stops']  # Customize!
candidate_metrics = [m for m in candidate_metrics if m not in exclude]
base_pairs = [
    # ('PredH_15T','clusT_total'),
    ('PredH_15T', 'RealH_15T'),
    ('PredHC_15T', 'RealHC_15T'),
    ('Pred', 'Real'),
    # ('PredH_15T','Stops'),
    # ('RealH_15T','Stops'),
    # ('RealH_15T','clusT_total')
]

# Time resolutions to substitute
time_resolutions = ['5T', '10T', '30T', '1H']

# Generate additional pairs
additional_pairs = []
for a, b in base_pairs:
    if '_15T' in a or '_15T' in b:
        for tr in time_resolutions:
            new_a = a.replace('_15T', f'_{tr}') if '_15T' in a else a
            new_b = b.replace('_15T', f'_{tr}') if '_15T' in b else b
            additional_pairs.append((new_a, new_b))

# Combine original and new
candidate_pairs = base_pairs + additional_pairs

refine = ['distance']

# Create all combinations of (a, b, c) where (a, b) is a matched pair and c comes from refine
all_combinations = [(a, b, c) for (a, b) in candidate_pairs for c in refine]

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
SET_SIZE = 2
addon = "Stops"
addon2 = "clusT_total"
addon3 = "Records"
addon4 = 'Uq'
addon5 = "Pred"
addon6 = "Real"
addon7 = "clusT_visits"
addon9 = "RealHC_15T"
addon10 = "PredHC_15T"
stacked = {}
stacked_max = {}


for xval, yval, zval in all_combinations:

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

        iou_scores = [(2 * x.loc[person]['IOU_GT'] * x.loc[person]['IOU_PRED']) /
                      (x.loc[person]['IOU_GT'] + x.loc[person]['IOU_PRED'] + 1e-6) for x in results.values()]
        annot = list(results.keys())

        # correlation = abs(np.corrcoef(x_values, y_values)[0, 1])
        # if correlation > 0.95:
        #     print(xval,yval,correlation,"OUT")
        #     continue

        addon_val = [x.loc[person][addon] for x in results.values()]
        addon2_val = [x.loc[person][addon2] for x in results.values()]
        addon3_val = [x.loc[person][addon3] for x in results.values()]
        addon4_val = [x.loc[person][addon4] for x in results.values()]
        addon5_val = [x.loc[person][addon5] for x in results.values()]
        addon6_val = [x.loc[person][addon6] for x in results.values()]
        addon7_val = [x.loc[person][addon7] for x in results.values()]
        addon9_val = [x.loc[person][addon9] for x in results.values()]
        addon10_val = [x.loc[person][addon10] for x in results.values()]
        data = pd.DataFrame({
            'x': x_values,
            'y': y_values,
            'iou': iou_scores,
            'annot': annot,
            f'{addon}': addon_val,
            f'{addon2}': addon2_val,
            f'{addon3}': addon3_val,
            f'{addon4}': addon4_val,
            f'{addon5}': addon5_val,
            f'{addon6}': addon6_val,
            f'{addon7}': addon7_val,
            f'{addon9}': addon9_val,
            f'{addon10}': addon10_val,
        })


        # Pareto front extraction
        # if xval == 'Stops':
        #     data['x'] = np.abs(data['x'].median() - data['x'])
        # if yval == 'Stops':
        #     data['y'] = np.abs(data['y'].median() - data['y'])
        data_sorted = data.reset_index(drop=True)
        points = data_sorted[['x', 'y']].values
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
        topN_in_pareto = [i for i in topN_idx if i in pareto_idx] # how many max are in pareto
        topN_in_pareto_prec = [i for i in topN_idx_prec if i in pareto_idx] # how many max are in pareto
        precision = len(topN_in_pareto_prec) / N_prec # if max is in pareto
        recall = 1 if len(topN_in_pareto) > 0 else 0 # if at least one of top 10 is in max
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

        #Score lift over basline
        numeric_cols = pareto_df.select_dtypes(include='number').columns
        lift = pareto_df['iou'].mean() - non_pareto_df['iou'].mean()
        lifts.append(lift)

        #Percentiles
        organised = data_sorted.sort_values('iou',ascending=False)
        organised['num'] = range(organised.shape[0])
        topsis = (organised[organised['pareto']].sort_values('num')['num'].iloc[0]+1)/organised.shape[0]
        percentile_ranks.append(topsis)

        pareto_sizes.append(pareto_df.shape[0]/data_sorted.shape[0])
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data_sorted.sort_values('iou')[['x', 'y', 'iou', 'pareto']])

        #IDEAL 1
        if 'clusT' in yval:
            ideal = [1.0, 1.0]
        if "Real" in yval:
            if 'T' in yval:
                multiplier = 60/int(''.join([w for w in yval if w.isdigit()]))
            else:
                multiplier = 1
            ideal = [1.0, np.log2(hours_per_person[person])*multiplier]
        data_sorted['distance'] = np.sqrt((data_sorted.x - ideal[0]) ** 2 + (data_sorted.y - ideal[1]) ** 2)
        min_dist = data_sorted[data_sorted.pareto].distance.idxmin()
        max_dist = data_sorted[data_sorted.pareto].distance.idxmax()
        min_dist_iou = data_sorted.iloc[min_dist].iou
        max_dist_iou = data_sorted.iloc[max_dist].iou
        stacked[person] = min_dist
        # stacked_max[person] = ((2*results["[50, 3, 50, '3T']"].IOU_GT * results["[50, 3, 50, '3T']"].IOU_PRED)/(results["[50, 3, 50, '3T']"].IOU_GT + results["[50, 3, 50, '3T']"].IOU_PRED))[person]
        stacked_max[person] = 0

        #IDEAL 2
        # ideal = [data_sorted.x.max(), data_sorted.y.max()]
        # data_sorted['distance'] = np.sqrt((data_sorted.x - ideal[0]) ** 2 + (data_sorted.y - ideal[1]) ** 2)
        # min_dist = data_sorted[zval].idxmin()
        # max_dist = data_sorted[zval].idxmax()
        # min_dist_iou = data_sorted.iloc[min_dist].iou
        # max_dist_iou = data_sorted.iloc[max_dist].iou

        # #IDEAL 2
        # ideal = [data_scaled[:, 0].max(), data_scaled[:, 1].max()]
        # data_sorted['distance'] = np.sqrt((data_scaled[:, 0] - ideal[0]) ** 2 + (data_scaled[:, 1] - ideal[1]) ** 2)
        # min_dist = data_sorted[data_sorted.pareto].distance.idxmin()
        # max_dist = data_sorted[data_sorted.pareto].distance.idxmax()
        # min_dist_iou = data_sorted.iloc[min_dist].iou
        # max_dist_iou = data_sorted.iloc[max_dist].iou

    combo_scores[(xval, yval, zval)] = str((np.mean(top3_scores),np.std(top3_scores)))
    precision_scores[(xval, yval, zval)] = np.mean(precisions)
    recall_scores[(xval, yval, zval )] = np.mean(recalls)
    hypervolumes[(xval, yval, zval )] = np.mean(hypervolume_vals)
    distances_to_best[(xval, yval, zval )] = np.mean(distance_vals)
    best_lift[(xval,yval, zval )] = np.mean(lifts)
    best_percentile_rank[(xval,yval, zval )] = np.mean(percentile_ranks)
    avg_pareto_sizes[(xval,yval, zval )] = np.mean(pareto_sizes)

    print(f"{xval}, {yval}, {zval} | Top-3 IOU: {combo_scores[(xval, yval, zval)]}, "
          f"P@{N_prec}: {precision_scores[(xval, yval, zval)]:.2f}, R@{N_recall}: {recall_scores[(xval, yval,zval)]:.2f}, "
          f"HV: {hypervolumes[(xval, yval,zval)]:.2f}, Dist: {distances_to_best[(xval, yval,zval)]:.2f},"
          f"Lift: {best_lift[(xval, yval,zval)]:.2f}, Percentile: {best_percentile_rank[(xval, yval,zval)]:.2f}, 'Size:': {avg_pareto_sizes[(xval, yval,zval)]:.2f}")
    print(f"Stacked avg:{np.mean(list(stacked.values()))}")
    print(f"Stacked max:{np.mean(list(stacked_max.values()))}")
combined = pd.concat([pd.DataFrame().from_dict(x,orient='index') for x in [combo_scores,precision_scores,recall_scores,hypervolumes,distances_to_best,best_lift,best_percentile_rank,avg_pareto_sizes]],axis=1)
combined.columns = ['TOP3','P','R','H','D','L','Perc','Size']
combined.sort_values('TOP3')
