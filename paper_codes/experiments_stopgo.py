import os
import pandas as pd
import geopandas as gpd
from stop_detection import run_infostop, stop_detection, ClusteringAggregator
from tqdm import tqdm
from predictability_metrics import *
from complexity_metrics import *
from sklearn.cluster import DBSCAN

tqdm.pandas()
# from humobi.measures.individual import real_predictability
import pandas as pd
from evaluation_metrics import *
from utils import remove1, nextstep, distances_pareto, distances, compare
from itertools import product
from pyproj import Transformer
import pickle
import numpy as np
import sys
sys.path.append('D:\GitHub\Stop-Go-Classifier')
from StopGoClassifier import StopGoClassifier
from sklearn.cluster import DBSCAN
from paper_codes.utils import *

# REFERENCE CALCULATIONS
# df = pd.read_csv('data/reference.csv').iloc[:, 1:]
df = pd.read_csv('synthetic_ground_truth/random_city.csv').iloc[:, 1:]
df['datetime'] = pd.to_datetime(df['datetime'])
df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lat,df.lon),crs=3857) #only synthetic
# df.columns = ['user_id', 'datetime', 'labels_reference', 'geometry', 'lon', 'lat']
df.columns = ['lat','lon','datetime','user_id','labels_reference','geometry']
tot_points = df.groupby('user_id').apply(lambda x: x.shape[0])
df = df.set_index('user_id')
df = df[['datetime', 'lon', 'lat', 'labels_reference']]

# NO PROCESSING
infostop_clus_time_total = clusters_time_entropy(df, 'labels_reference', scaling_type='Total')
infostop_clus_time_visits = clusters_time_entropy(df, 'labels_reference', scaling_type='Visits')

df_mod = df.rename({'labels_reference': 'labels'}, axis=1)

# HOURLY PROCESSING
SLOT = "15min"
slot_ns = pd.Timedelta(SLOT).value  # 900 s  → 9.0e+11 ns
slot_half = slot_ns // 2  # handy for tiebreaks

to_conca = {}
df_mod_grouped = df_mod.groupby(level=0, sort=False)
for uid, g in tqdm(df_mod_grouped,total=len(df_mod_grouped)):
    res = stays_to_slots_longest_fast(g, slot_ns=slot_ns)
    if res is not None:
        to_conca[uid] = res

# Combine all processed individual datasets back into a single DataFrame
df_hourly = pd.DataFrame(pd.concat(to_conca))
df_hourly = df_hourly.droplevel(1)
infostop_pred_hourly, infostop_real_hourly = real_predictability(df_hourly)

hourly_for_stats = df_hourly[df_hourly.labels != -1]
infostop_pred_hourly_clean, infostop_real_hourly_clean = real_predictability(hourly_for_stats)
# HOURLY END
# Next-step processing
df_cal = df[df.labels_reference != -1]
df_cal = nextstep(df_cal, 'labels_reference')
infostop_clus = clusters_entropy(df, tot_points, 'labels_reference')

# Properties
uq_points = df_cal.groupby(level=0).nunique().labels_reference
records = df.groupby(level=0).count().labels_reference
stops = df_cal.groupby(level=0).apply(lambda x: (x.labels_reference != x.labels_reference.shift()).sum())
df_resampled = df_cal.rename({'labels_reference': 'labels'}, axis=1)
infostop_pred, infostop_real = real_predictability(df_resampled)

reference = pd.concat(
    [infostop_pred, infostop_real, infostop_pred_hourly, infostop_real_hourly,
     infostop_clus_time_total, infostop_clus_time_visits, uq_points,
     records, stops,
     infostop_pred_hourly_clean, infostop_real_hourly_clean], axis=1)
reference.columns = ['Pred', 'Real', 'PredH', 'RealH', 'clusT_total',
                     'clusT_visits', 'Uq', 'Records', 'Stops', 'PredHC', "RealHC"]
df = df.reset_index()
df = df.sort_values(['user_id', 'datetime'])

# PREPARE STOP DETECTION
# param_dict = {
#     'MIN_STOP_INTERVAL': [5*60, 10*60, 15*60],
#     'MIN_DISTANCE_BETWEEN_STOP': [50, 100, 200],
#     'MIN_TIME_BETWEEN_STOPS': [3*60, 5*60, 10*60, 15*60],
#     'MAX_TIME_BETWEEN_STOPS_FOR_MERGE': [15*60, 60*60],
#     'eps': [100, 200]
# }
param_dict = {
    'MIN_STOP_INTERVAL': [5*60, 10*60, 15*60],
    'MIN_DISTANCE_BETWEEN_STOP': [5, 10, 20, 30, 50],
    'MIN_TIME_BETWEEN_STOPS': [3*60, 5*60, 10*60, 15*60],
    'MAX_TIME_BETWEEN_STOPS_FOR_MERGE': [15*60, 60*60],
    'eps': [10, 30, 50]
}

param_combinations = [
    dict(zip(param_dict.keys(), values))
    for values in product(*param_dict.values())
]
results = {}
# transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
# df[['lon', 'lat']] = df.apply(
#     lambda row: transformer.transform(row['lon'], row['lat']), axis=1,
#     result_type='expand')

# BEGIN EXPERIMENTS
for x in param_combinations:
    aggregated_df = {}
    totnumpoint = df.groupby('user_id').apply(lambda x: x.shape[0])
    classifier = StopGoClassifier(overwrite_settings = {k: v for k, v in x.items() if k != 'eps'})
    dbscan = DBSCAN(eps=x['eps'],min_samples=1)
    # df['unix'] = (df.datetime - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    for uid, udata in df.groupby('user_id'):
        udata = udata.copy()
        classifier.read(udata['datetime'].to_numpy(), udata['lat'].to_numpy(), udata['lon'].to_numpy())
        lbls = classifier.run()
        udata['labels'] = lbls.fillna(method='ffill').fillna(method='bfill')
        clustered_mask = udata['labels'] != -1
        if clustered_mask.sum() > 5:
            dj_udata = udata[clustered_mask]
            dj_lbls = dbscan.fit(dj_udata[['lat', 'lon']]).labels_
            udata.loc[clustered_mask, 'labels'] = dj_lbls
        aggregated_df[uid] = udata
        # udata = gpd.GeoDataFrame(udata, geometry=gpd.points_from_xy(udata['lon'], udata['lat']))
        # udata['datetime'] = udata.datetime.astype(str)
        # udata.to_file(f'outputs\stopgo_{uid}_{x.values()}.shp', driver='ESRI Shapefile')

    aggregated_df = pd.concat(aggregated_df).reset_index(drop=True)[
        ['user_id', 'datetime', 'lon', 'lat', 'labels_reference', 'labels']]
    aggregated_df.set_index('user_id', inplace=True)

    stop_overlap = overlap_fast(aggregated_df, 0.8)
    over = oversegmentation_fast(aggregated_df)
    under = undersegmentation_fast(aggregated_df)
    miss = missed_fast(aggregated_df)
    overde = overdetected_fast(aggregated_df)
    #
    infostop_clus_time_total = clusters_time_entropy(aggregated_df, 'labels', scaling_type='Total')
    infostop_clus_time_visits = clusters_time_entropy(aggregated_df, 'labels', scaling_type='Visits')
    # PROCESSING DATA
    SLOT_LIST = ['5min', '10min', '15min', '30min', '1H']
    # HOURLY PROCESSING
    # With this:
    hourly_metrics = {}

    for slot in SLOT_LIST:
        slot_ns = pd.Timedelta(slot).value
        to_conca = {}
        for uid, g in aggregated_df.groupby(level=0, sort=False):
            res = stays_to_slots_longest_fast(g, slot_ns=slot_ns)
            if res is not None:
                to_conca[uid] = res

        if not to_conca:
            continue  # Skip if empty

        df_slot = pd.concat(to_conca).droplevel(1)
        pred_h, real_h = real_predictability(df_slot)
        hourly_metrics[f'PredH_{slot}'] = pred_h
        hourly_metrics[f'RealH_{slot}'] = real_h

        # Clean version (non -1 labels)
        df_slot_clean = df_slot[df_slot.labels != -1]
        pred_hc, real_hc = real_predictability(df_slot_clean)
        hourly_metrics[f'PredHC_{slot}'] = pred_hc
        hourly_metrics[f'RealHC_{slot}'] = real_hc
    # HOURLY END
    # Next-step processing
    aggregated_df = aggregated_df[aggregated_df.labels != -1]
    aggregated_df = nextstep(aggregated_df, 'labels')

    uq_points = aggregated_df.groupby(level=0).nunique().labels
    records = aggregated_df.groupby(level=0).count().labels
    stops = aggregated_df.groupby(level=0).apply(lambda x: (x.labels_reference != x.labels_reference.shift()).sum())

    infostop_pred, infostop_real = real_predictability(aggregated_df)

    main_metrics = [infostop_pred, infostop_real, infostop_clus_time_total,
                    infostop_clus_time_visits, uq_points, records, stops,
                    miss, over, under, over['overseg'] / under['underseg'], overde]

    main_names = ['Pred', 'Real', 'clusT_total', 'clusT_visits', 'Uq',
                  'Records', "Stops", 'Miss', 'over', 'under', 'ouratio', 'overde']

    # Merge all metrics together
    summed = pd.concat(main_metrics + list(hourly_metrics.values()), axis=1)
    summed.columns = main_names + list(hourly_metrics.keys())

    summed = pd.concat((summed, stop_overlap), axis=1)
    summed = summed.sort_index().fillna(0)
    results[str([z for z in x.values()])] = summed
with open('results_stopgo_syn.pkl', 'wb') as f:
    pickle.dump(results, f)
with open('reference_stopgo_syn.pkl', 'wb') as f:
    pickle.dump(reference, f)
results
