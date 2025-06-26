import os
import geopandas as gpd
from stop_detection import infostop, stop_detection, ClusteringAggregator
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


def _explode_with_coverage(start_ns, end_ns):
    """
    Vector-explodes one stay [start, end) into all slot-ids it touches
    and returns: slot_id array, coverage array (in ns).
    """
    sid_first = start_ns // slot_ns
    sid_last = (end_ns - 1) // slot_ns  # inclusive
    n_slots = sid_last - sid_first + 1

    slot_ids = np.arange(n_slots, dtype=np.int64) + sid_first

    # coverage for each slot
    cov = np.empty(n_slots, dtype=np.int64)
    if n_slots == 1:  # stay fits in one slot
        cov[0] = end_ns - start_ns
    else:
        cov[0] = (sid_first + 1) * slot_ns - start_ns  # partial 1st
        cov[-1] = end_ns - sid_last * slot_ns  # partial last
        if n_slots > 2:
            cov[1:-1] = slot_ns  # full interior

    return slot_ids, cov


def stays_to_slots_longest(group, slot_ns, min_slot_coverage_ratio=0.5):
    """
    Assign slots to the label that covers the most duration in that slot.
    Handles -1 labels appropriately and avoids overextending label stays.
    """
    group = group.sort_values('datetime').drop_duplicates('datetime').reset_index(drop=True)
    if group.labels.nunique() < 2:
        return None

    group['label_change'] = (group.labels != group.labels.shift()).cumsum()

    slot_label_durations = {}
    slot_label_coords = {}

    for i, (_, sub) in enumerate(group.groupby('label_change')):
        label = sub.labels.iloc[0]
        start_ns = sub.datetime.iloc[0].value

        # Try to get the next timestamp globally (not just same label group)
        if sub.index[-1] + 1 < len(group):
            end_ns = group.iloc[sub.index[-1] + 1].datetime.value
        else:
            end_ns = sub.datetime.iloc[-1].value + slot_ns  # only pad if no more data

        sid_list, cov_list = _explode_with_coverage(start_ns, end_ns)
        lat = sub.lat.iloc[0]
        lon = sub.lon.iloc[0]

        for sid, cov in zip(sid_list, cov_list):
            key = (sid, label)
            slot_label_durations[key] = slot_label_durations.get(key, 0) + cov
            if key not in slot_label_coords:
                slot_label_coords[key] = (lat, lon)

    # Group by slot and select best label
    from collections import defaultdict
    slot_grouped = defaultdict(list)
    for (sid, label), dur in slot_label_durations.items():
        coord = slot_label_coords[(sid, label)]
        slot_grouped[sid].append((label, dur, coord))

    min_ns_covered = int(slot_ns * min_slot_coverage_ratio)
    rows = []

    for sid, label_data in slot_grouped.items():
        valid = [(l, d, c) for l, d, c in label_data if d >= min_ns_covered and l != -1]

        if valid:
            l, d, (lat, lon) = max(valid, key=lambda x: x[1])
        else:
            minus1s = [item for item in label_data if item[0] == -1]
            if not minus1s:
                continue
            l, d, (lat, lon) = max(minus1s, key=lambda x: x[1])

        rows.append({
            'datetime': pd.to_datetime(sid * slot_ns),
            'lat': lat,
            'lon': lon,
            'labels': l
        })

    return pd.DataFrame(rows)


# Resample by 1-hour interval and select the label with the longest duration
def longest_visited_row(groupa):
    """ Returns the row where the individual spent the longest time in an interval. """
    if groupa.empty:
        return pd.Series(dtype=object)  # Ensure an empty series is returned

    max_label = groupa.groupby('labels')['duration'].sum().idxmax()  # Find label with longest total duration

    # Select first row where this label appears
    row = groupa[groupa['labels'] == max_label].iloc[0]

    return row.T  # Return full row


# REFERENCE CALCULATIONS
df = pd.read_csv('data/reference.csv').iloc[:, 1:]
df['datetime'] = pd.to_datetime(df['datetime'])
df.columns = ['user_id', 'datetime', 'labels_reference', 'geometry', 'lon', 'lat']
tot_points = df.groupby('user_id').apply(lambda x: x.shape[0])
df = df.set_index('user_id')
df = df[['datetime', 'lon', 'lat', 'labels_reference']]

# NO PROCESSING
infostop_clus_time_total = clusters_time_entropy(df, 'labels_reference', scaling_type='Total')
infostop_clus_time_visits = clusters_time_entropy(df, 'labels_reference', scaling_type='Visits')

df_mod = df.rename({'labels_reference': 'labels'}, axis=1)


# HOURLY PROCESSING
SLOT = "15T"
slot_ns = pd.Timedelta(SLOT).value  # 900 s  → 9.0e+11 ns
slot_half = slot_ns // 2  # handy for tiebreaks

to_conca = {}
for uid, g in df_mod.groupby(level=0, sort=False):  # sort=False saves ~5 %
    res = stays_to_slots_longest(g, slot_ns=slot_ns)
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

# PREPARE STOP DETECION
param_dict = {
    'r1_level': [30, 50, 100, 200, 500, 1000],
    'min_staying_time': [60, 2 * 60, 3 * 60, 5 * 60, 10 * 60, 15 * 60, 20 * 60],
    'r2_level': [30, 50, 100, 300, 500]
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
    totnumpoint = df.groupby('user_id').apply(lambda x: x.shape[0])
    infostop_df = infostop(df, r1=x['r1_level'], r2=x['r2_level'], min_staying_time=x['min_staying_time'])
    aggregated_df = pd.concat([infostop_df, df.set_index('user_id').add_suffix('_2')], axis=1)
    aggregated_df = aggregated_df[['lat', 'lon', 'datetime', 'labels', 'labels_reference_2']]
    aggregated_df.rename({'labels_reference_2': 'labels_reference'}, axis=1, inplace=True)
    aggregated_df.index.set_names('user_id', inplace=True)
    for uid, udata in aggregated_df.groupby(level=0):
        udata = gpd.GeoDataFrame(udata, geometry=gpd.points_from_xy(udata['lon'], udata['lat']))
        udata['datetime'] = udata.datetime.astype(str)
        udata.to_file(f'outputs\infostop_{uid}_{x.values()}.shp', driver='ESRI Shapefile')

    stop_overlap = overlap(aggregated_df, 0.8)
    over = oversegmentation(aggregated_df)
    under = undersegmentation(aggregated_df)
    miss = missed(aggregated_df)
    overde = overdetected(aggregated_df)
    #
    infostop_clus_time_total = clusters_time_entropy(aggregated_df, 'labels', scaling_type='Total')
    infostop_clus_time_visits = clusters_time_entropy(aggregated_df, 'labels', scaling_type='Visits')

    # PROCESSING DATA
    SLOT_LIST = ['5T', '10T', '15T', '30T', '1H']
    # HOURLY PROCESSING
    # With this:
    hourly_metrics = {}

    for slot in SLOT_LIST:
        slot_ns = pd.Timedelta(slot).value
        to_conca = {}
        for uid, g in aggregated_df.groupby(level=0, sort=False):
            res = stays_to_slots_longest(g, slot_ns=slot_ns)
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
                    miss, over, under, over / under, overde]

    main_names = ['Pred', 'Real', 'clusT_total', 'clusT_visits', 'Uq',
                  'Records', "Stops", 'Miss', 'over', 'under', 'ouratio', 'overde']

    # Merge all metrics together
    summed = pd.concat(main_metrics + list(hourly_metrics.values()), axis=1)
    summed.columns = main_names + list(hourly_metrics.keys())

    summed = pd.concat((summed, stop_overlap), axis=1)
    summed = summed.sort_index().fillna(0)
    results[str([z for z in x.values()])] = summed
with open('results_infostop.pkl', 'wb') as f:
    pickle.dump(results, f)
with open('reference_infostop.pkl', 'wb') as f:
    pickle.dump(reference, f)
results
