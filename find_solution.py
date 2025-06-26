import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly
import seaborn as sns

def compare(reference, results, column_to):
    closest = pd.concat([x.idxmin() for x in distances(reference,results,column_to)]).reset_index(drop=True)
    best_clustering = pd.concat([x['IOU_GT'] for x in results.values()],axis=1).T.reset_index(drop=True).T.idxmax(axis=1)
    best_ouratio = pd.concat([abs(x['ouratio']-1) for x in results.values()],axis=1).T.reset_index(drop=True).T.idxmin(axis=1)
    best_ratio = pd.concat([np.sqrt(x[column_to]*x.Pred) for x in results.values()],axis=1).T.reset_index(drop=True).T.idxmax(axis=1)
    pareto_distance = pd.concat([x.idxmin() for x in distances_pareto(reference,results,column_to)]).reset_index(drop=True)
    stacked = pd.concat([closest,best_clustering,best_ouratio,best_ratio,pareto_distance],axis=1)
    stacked.columns = ['closest','eval','ouratio','ratio','pareto']
    return stacked

def distances_pareto(reference, results, col_name):
    max_pred = pd.concat([x.Pred for x in results.values()],axis=1).max(axis=1)
    max_col = pd.concat([x[col_name] for x in results.values()],axis=1).max(axis=1)
    res = []
    for uid,da in reference.groupby(level=0):
        res.append(pd.DataFrame(np.sqrt((x.loc[uid].Pred - max_pred.loc[uid]) ** 2 + (x.loc[uid][col_name] - max_col.loc[uid]) ** 2) for x in results.values()))
    return res


def distances(reference, results, col_name):
    res = []
    for uid,da in reference.groupby(level=0):
        res.append(pd.DataFrame(np.sqrt((x.loc[uid].Pred - reference.loc[uid].Pred) ** 2 + (x.loc[uid][col_name] - reference.loc[uid][col_name]) ** 2) for x in results.values()))
    return res

def is_pareto_efficient(costs):
    """
    Find the Pareto-efficient points in a 3D objective space.
    Args:
        costs (np.ndarray): Array of shape (n_points, 3) with each row as (x, y, z)
    Returns:
        np.ndarray: Boolean mask of Pareto-efficient points.
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] > c, axis=1) | np.all(costs[is_efficient] == c, axis=1)
            is_efficient[i] = True  # Keep self
    return is_efficient



# Path to the .pkl file
res = 'results_dbscan.pkl'

# Load the dictionary from the .pkl file
with open(res, 'rb') as file:
    results = pickle.load(file)


# Path to the .pkl file
ref = 'reference_dbscan.pkl'

# Load the dictionary from the .pkl file
with open(ref, 'rb') as file:
    reference = pickle.load(file)

top_pareto_summary = {}
xval = "Pred"
yval = "clusT_total"
zval = "Stops"

for person in reference.index:
    x_values = [x.loc[person][xval] for x in results.values()]
    y_values = [x.loc[person][yval] for x in results.values()]
    z_values = [x.loc[person][zval] for x in results.values()]
    iou_scores = [(2 * x.loc[person]['IOU_GT'] * x.loc[person]['IOU_PRED']) /
                  (x.loc[person]['IOU_GT'] + x.loc[person]['IOU_PRED']) for x in results.values()]
    annot = list(results.keys())

    data = pd.DataFrame({
        'x': x_values,
        'y': y_values,
        'z': z_values,
        'iou': iou_scores,
        'annot': annot
    })

    # Determine Pareto-efficient rows
    pareto_mask = is_pareto_efficient(data[['x', 'y', 'z']].values)
    data['pareto'] = pareto_mask

    pareto_df = data[data['pareto']].sort_values('iou', ascending=False).reset_index(drop=True)
    top3 = pareto_df.head(3)

    top_pareto_summary[person] = {
        'top1_iou': top3.iloc[0]['iou'],
        'top3_mean_iou': top3['iou'].mean(),
        'pareto_max_iou': pareto_df['iou'].max(),
        'pareto_mean_iou': pareto_df['iou'].mean(),
        'non_pareto_mean_iou': data[~data['pareto']]['iou'].mean(),
        'non_pareto_max_iou': data[~data['pareto']]['iou'].max(),
        'non_pareto_top3_mean': data[~data['pareto']]['iou'].sort_values().tail(3).mean(),
        'complexity_top1': top3.iloc[0]['y'],
        'complexity_top3_mean': top3['y'].mean(),
        'distance_to_best': data['iou'].max() - top3['iou'].max()
    }

top_pareto_df = pd.DataFrame.from_dict(top_pareto_summary, orient='index')
print((top_pareto_df['pareto_mean_iou'] - top_pareto_df['non_pareto_mean_iou']).mean())
top_pareto_df
