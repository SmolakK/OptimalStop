import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.preprocessing import MinMaxScaler
from results_utils import pareto_mask, global_sensitivity_analysis
from tqdm import tqdm

def compute_iou(row):
    """Compute harmonic mean between predicted and GT IOUs."""
    return (2 * row['IOU_GT'] * row['IOU_PRED']) / (row['IOU_GT'] + row['IOU_PRED'] + 1e-6)


def load_reference_data(path: str) -> gpd.GeoDataFrame:
    df = pd.read_csv(path).iloc[:, 1:]
    df['datetime'] = pd.to_datetime(df['datetime'])

    if 'synthetic' in path:
        df = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.lat, df.lon), crs=3857
        )
        df.columns = ['lat', 'lon', 'datetime', 'user_id', 'labels_reference', 'geometry']
    else:
        df.columns = ['user_id', 'datetime', 'labels_reference', 'geometry', 'lon', 'lat']
    return df


def load_pickle_dict(path: str) -> dict:
    with open(path, 'rb') as f:
        data = pd.read_pickle(f)
    return data


def generate_candidate_pairs(candidate_metrics: list) -> list:
    if any('_15min' in m for m in candidate_metrics):
        base_pairs = [
            ('PredH_15min', 'RealH_15min'),
            ('PredHC_15min', 'RealHC_15min'),
            ('Pred', 'Real'),
        ]
        time_resolutions = ['5min', '10min', '30min', '1H']
        tag = '_15min'
    else:
        base_pairs = [
            ('PredH_15T', 'RealH_15T'),
            ('PredHC_15T', 'RealHC_15T'),
            ('Pred', 'Real'),
        ]
        time_resolutions = ['5T', '10T', '30T', '1H']
        tag = '_15T'

    additional_pairs = [
        (a.replace(tag, f'_{tr}') if tag in a else a,
         b.replace(tag, f'_{tr}') if tag in b else b)
        for a, b in base_pairs
        for tr in time_resolutions
    ]
    return base_pairs + additional_pairs


def results_analysis(reference_path, ref_pkl, res_pkl):
    """Comprehensive evaluation of stop detection quality on Pareto vs non-Pareto results."""
    # === Load data ===
    df = load_reference_data(reference_path)
    hours_per_person = df.groupby('user_id')['datetime'].apply(
        lambda x: (x.max() - x.min()).total_seconds() / 3600
    )
    reference = load_pickle_dict(ref_pkl)
    results = load_pickle_dict(res_pkl)

    # === Prepare metrics ===
    sample_result = next(iter(results.values()))
    exclude = {'IOU_GT', 'IOU_PRED', 'eval', 'ouratio', 'Miss', 'over', 'under', 'overde',
               'VI', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1',
               'StopsH', 'RecordsH', 'UqH', 'clus_points', 'clus_stops'}
    candidate_metrics = [m for m in sample_result.columns if m not in exclude]
    candidate_pairs = generate_candidate_pairs(candidate_metrics)

    quality_metrics = ['Miss', 'over', 'under', 'ouratio', 'overde']
    structure_metrics = ['Stops', 'Uq', 'Records']

    refine = ['distance']
    combinations = [(a, b, c) for (a, b) in candidate_pairs for c in refine]
    combinations = list(set(combinations))
    results = {k: v for k, v in results.items() if v.shape[1] >= 34}

    summary_stats = []

    ## GET SENSITIVITY ANALYSIS
    sensitivity_pick = global_sensitivity_analysis(results)

    for xval, yval, zval in combinations:
        pareto_summary, best_summary, sensitivity_summary = [],[],[]

        for person in tqdm(reference.index,total=reference.shape[0]):
            person_scores = {}
            try:
                df_person = pd.DataFrame({
                    'x': [res.loc[person][xval] for res in results.values()],
                    'y': [res.loc[person][yval] for res in results.values()],
                    'iou': [compute_iou(res.loc[person]) for res in results.values()],
                    **{m: [res.loc[person][m] for res in results.values()]
                       for m in quality_metrics + structure_metrics}
                }, index=results.keys())

                pareto_mask_ = pareto_mask(df_person[['x', 'y']].values)
                df_person['pareto'] = pareto_mask_
                pareto_df = df_person[df_person['pareto']]
                if pareto_df.empty:
                    continue

                data_sorted = df_person.sort_values('iou', ascending=False)
                # TOP3 of scores
                top3 = data_sorted[data_sorted['pareto']].head(10)
                top3_iou = top3['iou'].mean()
                person_scores['top10_iou'] = top3_iou

                # Precision and recall
                N_prec = 1
                N_recall = 10
                topN_idx = data_sorted.head(N_recall).index
                topN_idx_prec = data_sorted.head(N_prec).index
                pareto_idx = data_sorted[data_sorted['pareto']].index
                topN_in_pareto = [i for i in topN_idx if i in pareto_idx]  # how many max are in pareto
                topN_in_pareto_prec = [i for i in topN_idx_prec if i in pareto_idx]  # how many max are in pareto
                precision = len(topN_in_pareto_prec) / N_prec  # if max is in pareto
                recall = 1 if len(topN_in_pareto) > 0 else 0  # if at least one of top 10 is in max
                person_scores['precision'] = precision
                person_scores['recall'] = recall

                # Distance to best IOUs
                best_total_iou = data_sorted['iou'].max()
                best_pareto_iou = data_sorted[data_sorted['pareto']]['iou'].max()
                dist = abs(best_total_iou - best_pareto_iou)
                person_scores['dist'] = dist

                # Pareto size
                pareto_size = df_person['pareto'].sum() / df_person.shape[0]  # how many solutions lie on pareto
                person_scores['size'] = pareto_size

                # Pick the best solution
                if "Real" in yval:
                    if 'min' in yval:
                        multiplier = 60 / int(''.join([w for w in yval if w.isdigit()]))
                    else:
                        multiplier = 1
                ideal = [1.0, np.log2(hours_per_person[person] * multiplier)]
                data_sorted['distance'] = np.sqrt((data_sorted.x - ideal[0]) ** 2 + (data_sorted.y - ideal[1]) ** 2)
                min_dist = data_sorted[data_sorted.pareto].distance.idxmin()
                min_dist_scores = data_sorted.loc[min_dist]

                pf = df_person[df_person['pareto']]

                # 2. Normalize x and y
                # pf.loc[:,'y'] = pf['y'] / np.log2(pf['Uq'])
                x_norm = (pf['x'].max() - pf['x']) / (pf['x'].max() - pf['x'].min() + 1e-9)
                y_norm = (pf['y'] - pf['y'].min()) / (pf['y'].max() - pf['y'].min() + 1e-9)

                # 3. Combined score: maximize x↓ + y↑
                score = x_norm + y_norm

                min_dist_scores = pf.loc[score.idxmax()]

                # Get sensitivity scores
                sensitivity_scores = df_person.loc[sensitivity_pick]

                pareto_summary.append(person_scores)
                best_summary.append(min_dist_scores)
                sensitivity_summary.append(sensitivity_scores)

            except Exception as e:
                print(f"Error processing user {person}: {e}")
                continue

        # Convert lists to DataFrames for summary aggregation
        pareto_summary = pd.DataFrame(pareto_summary)
        best_summary = pd.DataFrame(best_summary)
        sensitivity_summary = pd.DataFrame(sensitivity_summary)

        best_summary = best_summary.loc[:, [x for x in best_summary.columns if x != 'pareto']]
        sensitivity_summary = sensitivity_summary.loc[:, [x for x in sensitivity_summary.columns if x != 'pareto']]

        stat_entry =  {
    'xval': xval, 'yval': yval, 'zval': zval,
    **{f'Pareto_mean_{m}': pareto_summary[m].mean() for m in pareto_summary.columns},
    **{f'Best_mean_{m}': best_summary[m].mean() for m in best_summary.columns if m not in ['x','y']},
    **{f'Sensitivity_mean_{m}': sensitivity_summary[m].mean() for m in sensitivity_summary.columns if m not in ['x','y']},
     **{f'Pareto_std_{m}': pareto_summary[m].std() for m in pareto_summary.columns},
     **{f'Best_std_{m}': best_summary[m].std() for m in best_summary.columns if m not in ['x', 'y']},
     **{f'Sensitivity_std_{m}': sensitivity_summary[m].std() for m in sensitivity_summary.columns if
        m not in ['x', 'y']},
}

        summary_stats.append(stat_entry)
        print(f"✓ {xval}, {yval}: Pareto vs All quality metrics summarized.")

    combined = pd.DataFrame(summary_stats)
    combined.set_index(['xval','yval','zval'],inplace=True)
    return combined


# === Example usage ===
if __name__ == "__main__":
    file_path = r"D:\GitHub\OptimalStop\synthetic_ground_truth\random_city.csv"
    ref_pkl = r"D:\GitHub\OptimalStop\reference_infostop_syn.pkl"
    res_pkl = r"D:\GitHub\OptimalStop\results_infostop_syn.pkl"
    combined_df = results_analysis(file_path, ref_pkl, res_pkl)
    # combined_df.to_csv(r'results_infostop_syn_std.csv')
    #
    ref_pkl = r"D:\GitHub\OptimalStop\reference_stopgo_syn.pkl"
    res_pkl = r"D:\GitHub\OptimalStop\results_stopgo_syn.pkl"
    combined_df = results_analysis(file_path, ref_pkl, res_pkl)
    # combined_df.to_csv(r'results_stopgo_syn_std.csv')
    #
    ref_pkl = r"D:\GitHub\OptimalStop\reference_dbscan_syn.pkl"
    res_pkl = r"D:\GitHub\OptimalStop\results_dbscan_syn.pkl"
    combined_df = results_analysis(file_path, ref_pkl, res_pkl)
    # combined_df.to_csv(r'results_dbscan_syn_std.csv')

    ref_pkl = r"D:\GitHub\OptimalStop\reference_stdbscan_syn.pkl"
    res_pkl = r"D:\GitHub\OptimalStop\results_stdbscan_syn_full.pkl"
    combined_df = results_analysis(file_path, ref_pkl, res_pkl)
    # combined_df.to_csv(r'results_stdbscan_syn_std.csv')

    # REAL DATA
    # file_path = r"D:\GitHub\OptimalStop\data\reference.csv"
    # ref_pkl = r"D:\GitHub\OptimalStop\reference_infostop.pkl"
    # res_pkl = r"D:\GitHub\OptimalStop\results_infostop.pkl"
    # combined_df = results_analysis(file_path, ref_pkl, res_pkl)
    # combined_df.to_csv(r'results_infostop_std.csv')
    #
    # ref_pkl = r"D:\GitHub\OptimalStop\reference_stopgo.pkl"
    # res_pkl = r"D:\GitHub\OptimalStop\results_stopgo.pkl"
    # combined_df = results_analysis(file_path, ref_pkl, res_pkl)
    # combined_df.to_csv(r'results_stopgo_std.csv')

    # ref_pkl = r"D:\GitHub\OptimalStop\reference_dbscan.pkl"
    # res_pkl = r"D:\GitHub\OptimalStop\results_dbscan.pkl"
    # combined_df = results_analysis(file_path, ref_pkl, res_pkl)
    # combined_df.to_csv(r'results_dbscan_std.csv')

    # ref_pkl = r"D:\GitHub\OptimalStop\reference_stdbscan.pkl"
    # res_pkl = r"D:\GitHub\OptimalStop\results_stdbscan.pkl"
    # combined_df = results_analysis(file_path, ref_pkl, res_pkl)
    # combined_df.to_csv(r'results_stdbscan_std.csv')
