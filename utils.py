from numba import cuda, jit
import numpy as np
from math import ceil
import pandas as pd


def compare(reference, results, column_to):
    closest = pd.concat([x.idxmin() for x in distances(reference,results,column_to)]).reset_index(drop=True)
    best_clustering = pd.concat([x['eval'] for x in results.values()],axis=1).T.reset_index(drop=True).T.idxmax(axis=1)
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


def mode_label(series):
    return series.mode()[0] if not series.empty else None


def remove1(trajectories, column_name):
    return trajectories[trajectories[column_name] != -1]


def nextstep(trajectories, column_name):
    mask = trajectories[column_name] != trajectories[column_name].shift()

    # 2. rows that genuinely represent transitions
    transitioned = trajectories[mask]

    # 3. users that vanished because they had no transitions
    all_users = trajectories.index.get_level_values(0).unique()
    surviving = transitioned.index.get_level_values(0).unique()
    no_transition = all_users.difference(surviving)

    if len(no_transition):
        # take the first row for each “constant-label” user
        first_rows = (trajectories
                      .loc[no_transition]
                      .groupby(level=0, group_keys=False)
                      .head(1))
        # add them back and keep the chronological order
        transitioned = pd.concat([transitioned, first_rows]).sort_index()

    return transitioned


@cuda.jit
def _matchfinder_gpu(gs, data_len, output):
    """
    Finds the shortest not repeating sequences according to the Lempel-Ziv algorithm. Algorithm adaptation for GPU.
    :param gs: symbol series
    :param data_len: data length
    :param output: output array
    """
    pos = cuda.grid(1)
    max_subsequence_matched = 0
    finish_bool = False
    if pos < data_len:
        for i in range(0, pos):
            j = 0
            end_distance = data_len - pos
            while (pos + j < data_len) and (i + j < pos) and (gs[i + j] == gs[pos + j]):
                j += 1
            if j == end_distance:
                finish_bool = True
                break
            elif j > max_subsequence_matched:
                max_subsequence_matched = j
        if finish_bool:
            output[pos] = end_distance + 1  # CHANGED with XU paper
        else:
            output[pos] = max_subsequence_matched + 1


def matchfinder(gs, gpu=True):
    """
    Finds the shortest not repeating sequences according to the Lempel-Ziv algorithm
    :param gs: symbol series
    :return: the length of the shortest non-repeating subsequences at each step of sequence
    """
    gs = gs.dropna()
    data_len = len(gs)
    gs_array = np.array(gs.values)
    output = np.zeros(data_len)
    output[0] = 1
    if gpu:
        # Ensure CUDA context is initialized
        cuda.current_context()

        # Transfer data to the device
        d_gs = cuda.to_device(gs_array)
        d_output = cuda.to_device(output)

        # Configure kernel launch parameters
        threadsperblock = 256
        blockspergrid = ceil(data_len / threadsperblock)

        # Launch te kernel
        _matchfinder_gpu[threadsperblock, blockspergrid](d_gs, data_len,d_output)

        # Copy results back to host
        output = d_output.copy_to_host()
    return output


def _fit_func(x, a, b, c):
    return a * np.exp(b * x) + c


def stays_to_slots_longest_fast(group, slot_ns, min_slot_coverage_ratio=0.5):
    """
    Vectorized version of stays_to_slots_longest.
    Assign each slot to the label covering the most time duration.
    """
    group = group.sort_values('datetime').drop_duplicates('datetime')
    if group.labels.nunique() < 2:
        return None

    datetimes = group['datetime'].values.astype('int64')
    labels = group['labels'].values
    lats = group['lat'].values
    lons = group['lon'].values

    # Compute start/end times of each stay
    change_idx = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    start_ns = datetimes[change_idx]
    end_ns = np.r_[datetimes[change_idx[1:]], datetimes[-1] + slot_ns]
    seg_labels = labels[change_idx]
    seg_lats = lats[change_idx]
    seg_lons = lons[change_idx]

    # Compute slot indices
    start_slots = start_ns // slot_ns
    end_slots = end_ns // slot_ns

    # Prepare arrays for accumulation
    slot_keys = []
    slot_labels = []
    slot_durations = []
    slot_coords = []

    min_ns_covered = int(slot_ns * min_slot_coverage_ratio)

    for s_slot, e_slot, l, st, en, lat, lon in zip(start_slots, end_slots, seg_labels, start_ns, end_ns, seg_lats, seg_lons):
        # Continuous slot range
        slots = np.arange(s_slot, e_slot + 1)
        # Duration overlap with each slot
        slot_start = slots * slot_ns
        slot_end = (slots + 1) * slot_ns
        overlap = np.clip(np.minimum(slot_end, en) - np.maximum(slot_start, st), 0, slot_ns)

        valid = overlap >= min_slot_coverage_ratio * slot_ns
        if not np.any(valid):
            # fallback for missing coverage: mark as -1
            slot_keys.extend(slots)
            slot_labels.extend([-1] * len(slots))
            slot_durations.extend(overlap)
            slot_coords.extend([(lat, lon)] * len(slots))
            continue

        slot_keys.extend(slots[valid])
        slot_labels.extend([l] * np.sum(valid))
        slot_durations.extend(overlap[valid])
        slot_coords.extend([(lat, lon)] * np.sum(valid))

    if not slot_keys:
        return None

    df = pd.DataFrame({
        'slot': slot_keys,
        'labels': slot_labels,
        'duration': slot_durations,
        'coord': slot_coords
    })

    # Select best label per slot
    df_best = (df.loc[df.groupby('slot')['duration'].idxmax()]
               .reset_index(drop=True))

    # Replace -1 slots if no valid labels exist
    df_best.loc[df_best['labels'] == -1, 'labels'] = -1

    out = pd.DataFrame({
        'datetime': pd.to_datetime(df_best['slot'] * slot_ns),
        'lat': [c[0] for c in df_best['coord']],
        'lon': [c[1] for c in df_best['coord']],
        'labels': df_best['labels']
    })

    return out


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
