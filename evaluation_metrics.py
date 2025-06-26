import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, \
    homogeneity_completeness_v_measure, pair_confusion_matrix, v_measure_score, completeness_score
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, f1_score
import numpy as np
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment


def overdetected(trajectories):
    users = {}
    for uid, group in trajectories.groupby(level=0):
        overdetect = group.groupby('labels').apply(lambda x: (x.labels_reference == -1).all()).sum()
        users[uid] = overdetect/len(group.groupby('labels'))
    return pd.DataFrame.from_dict(users,orient='index')


def missed(trajectories):
    users = {}
    trajectories = trajectories[trajectories.labels_reference != -1]
    for uid, group in trajectories.groupby(level=0):
        missed = group.groupby('labels_reference').apply(lambda x: (x.labels == -1).all()).sum()
        users[uid] = missed/len(group.groupby('labels_reference'))
    return pd.DataFrame.from_dict(users,orient='index')


def _variation_of_information(labels_true, labels_pred):
    """
    Compute the Variation of Information (VI) between two clusterings:
    - VI = H(C|K) + H(K|C)
    where:
        H(C|K) = H(C,K) - H(K)  (under-segmentation / merging)
        H(K|C) = H(C,K) - H(C)  (over-segmentation / splitting)

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,)
        Ground-truth cluster labels.
    labels_pred : array-like of shape (n_samples,)
        Predicted cluster labels.

    Returns
    -------
    VI : float
        The total Variation of Information.
    H_C_given_K : float
        The conditional entropy H(C|K), measuring under-segmentation.
    H_K_given_C : float
        The conditional entropy H(K|C), measuring over-segmentation.
    """

    # Build the contingency matrix (rows = true clusters, cols = predicted clusters)
    cm = contingency_matrix(labels_true, labels_pred)
    n = cm.sum()  # total number of samples

    # Convert counts to probabilities p(c,k)
    p_ck = cm / n

    # Marginal distributions p(c) and p(k)
    p_c = p_ck.sum(axis=1)  # sum over columns
    p_k = p_ck.sum(axis=0)  # sum over rows

    # Entropies: H(C), H(K), H(C,K)
    def entropy(p):
        return -np.sum(p[p > 0] * np.log2(p[p > 0]))

    H_C = entropy(p_c)
    H_K = entropy(p_k)
    H_CK = entropy(p_ck.flatten())  # flatten to treat joint as 1D distribution

    # Conditional entropies
    # H(C|K) = H(C,K) - H(K)
    H_C_given_K = H_CK - H_K
    # H(K|C) = H(C,K) - H(C)
    H_K_given_C = H_CK - H_C

    # Variation of Information
    VI = H_C_given_K + H_K_given_C

    return VI, H_C_given_K, H_K_given_C


def variation_of_information(trajectories):
    users = {}
    for uid, group in trajectories.groupby(level=0):
        vi, under_seg, over_seg =_variation_of_information(group.labels_reference,group.labels)
        users[uid] = [vi, under_seg, over_seg]
    users = pd.DataFrame(users).T
    users.columns = ['VI','under','over']
    return users


def undersegmentation(trajectories):
    users = {}
    for uid, group in trajectories.groupby(level=0):
        group = group.reset_index(drop=True)
        group['visits'] = group.labels != group.labels.shift()
        group['stops'] = group.visits.cumsum()
        total_segments = 0
        total_stops = 0
        for stop_id, stopdata in group.groupby('stops'):
            if not (stopdata.labels == -1).all() and not (stopdata.labels_reference == -1).all():
                total_segments += ((stopdata.labels_reference != stopdata.labels_reference.shift()) & (stopdata.labels_reference != -1)).sum()
                total_stops += 1
        if total_stops == 0:
            users[uid] = 0
        else:
            users[uid] = total_segments/total_stops
    return pd.DataFrame().from_dict(users,orient='index')


def oversegmentation(trajectories):
    users = {}
    trajectories = trajectories[trajectories.labels_reference != -1]
    for uid, group in trajectories.groupby(level=0):
        group = group.reset_index(drop=True)
        group['visits'] = group.labels_reference != group.labels_reference.shift()
        group['stops'] = group.visits.cumsum()
        total_segments = 0
        total_stops = 0
        for stop_id, stopdata in group.groupby('stops'):
            if not (stopdata.labels == -1).all():
                first = (stopdata.labels != -1).idxmax()
                last = (stopdata.labels[::-1] != -1).idxmax()
                stopdata = stopdata.loc[first:last]
            detected_segments = (stopdata.labels != stopdata.labels.shift()).sum()
            total_segments += detected_segments
            total_stops += 1
        users[uid] = total_segments/total_stops
    return pd.DataFrame().from_dict(users,orient='index')


def clustering_evaluation(trajectories, single_point=True):
    processed = {}
    for uid, group in trajectories.groupby(level=0):
        processed_uid = []
        group['moved'] = (group.labels != group.labels.shift()).cumsum()
        for lbl, stopdata in group.groupby('moved'):
            if not (stopdata.labels == stopdata.labels_reference).all():
                separated = stopdata.groupby(['labels','labels_reference']).first().reset_index()
                processed_uid.append(separated)
            else:
                processed_uid.append(pd.DataFrame(stopdata.iloc[0]).T)
        processed_uid = pd.concat(processed_uid)
        processed[uid] = adjusted_rand_score(processed_uid.labels_reference,processed_uid.labels)
    return pd.DataFrame.from_dict(processed,orient='index')

def get_time_overlap(row):
    """
    Given a row containing:
      - start_time, end_time (predicted)
      - gt_start_time, gt_end_time (ground truth)
    returns the duration of their overlap as a pandas Timedelta.
    """
    overlap_start = max(row["start_time"], row["gt_start_time"])
    overlap_end   = min(row["end_time"],   row["gt_end_time"])
    return max(pd.Timedelta(0), overlap_end - overlap_start)


def interval_iou(startA, endA, startB, endB):
    """
    Computes the Intersection-over-Union (IoU) for two time intervals:
    [startA, endA] and [startB, endB].
    """
    # Convert to timestamps (if needed)
    # startA, endA, startB, endB = pd.to_datetime(startA), pd.to_datetime(endA), ...

    overlap_start = max(startA, startB)
    overlap_end = min(endA, endB)

    # If there's no overlap, IoU = 0
    overlap = (overlap_end - overlap_start).total_seconds()
    if overlap < 0:
        overlap = 0  # No overlap

    union = (endA - startA).total_seconds() + (endB - startB).total_seconds() - overlap
    if union <= 0:
        return 0

    return overlap / union


def overlap(trajectories, threshold=0.5):
    traj_stops = trajectories[trajectories['labels'] != -1].copy().reset_index()
    traj_stops.sort_values(["user_id", "datetime"], inplace=True)
    traj_stops['moved'] = traj_stops.groupby(by='user_id').apply(lambda x: (x['labels'] != x['labels'].shift()).cumsum()).droplevel(0)
    det_stop_intervals = (
        traj_stops
            .groupby(["user_id", "moved"], as_index=False)
            .agg(
            start_time=("datetime", "first"),
            end_time=("datetime", "last"),
        )
    )

    traj_stops = trajectories[trajectories['labels_reference'] != -1].copy().reset_index()
    traj_stops.sort_values(["user_id", "datetime"], inplace=True)
    traj_stops['moved'] = traj_stops.groupby(by='user_id').apply(
        lambda x: (x['labels_reference'] != x['labels_reference'].shift()).cumsum()).droplevel(0)
    true_stop_intervals = (
        traj_stops
            .groupby(["user_id", "moved"], as_index=False)
            .agg(
            gt_start_time=("datetime", "first"),
            gt_end_time=("datetime", "last"),
        )
    )
    all_users = true_stop_intervals.user_id.unique()
    time_metrics = []
    for user in all_users:
        pred_user = det_stop_intervals[det_stop_intervals['user_id'] == user].copy()
        gt_user = true_stop_intervals[true_stop_intervals['user_id'] == user].copy()
        if pred_user.empty:
            time_metrics.append((user, 0, 0))
            continue
        # 0) Add duration
        pred_user['duration'] = (pred_user['end_time'] - pred_user['start_time']).dt.total_seconds()
        gt_user['duration'] = (gt_user['gt_end_time'] - gt_user['gt_start_time']).dt.total_seconds()


        # 1) Cross join:
        pred_user["key"] = 1
        gt_user["key"] = 1
        merged = pd.merge(pred_user, gt_user, on="key")
        merged.drop("key", axis=1, inplace=True)

        merged["iou"] = merged.apply(
            lambda row: interval_iou(
                row["start_time"],
                row["end_time"],
                row["gt_start_time"],
                row["gt_end_time"]
            ), axis=1
        )

        best_iou_per_pred = merged.groupby(['user_id_x', 'moved_x', "duration_x"])["iou"].max().reset_index()
        precision_time = (best_iou_per_pred['duration_x'] * best_iou_per_pred['iou']).sum()/best_iou_per_pred['duration_x'].sum()

        # 4) For ground truth: we want to see how many intervals got matched.
        best_iou_per_gt = merged.groupby(["user_id_y", "moved_y", "duration_y"])["iou"].max().reset_index()
        recall_time = (best_iou_per_gt['duration_y'] * best_iou_per_gt['iou']).sum()/best_iou_per_gt['duration_y'].sum()

        time_metrics.append((user, precision_time, recall_time))
    time_metrics_df = pd.DataFrame(
    time_metrics,
    columns=["user_id",'IOU_PRED','IOU_GT']
)
    time_metrics_df.set_index('user_id',inplace=True)
    return time_metrics_df


def match_intervals_bipartite(trajectories, iou_threshold=0.5):
    """
    Performs bipartite matching for the intervals in user_pred vs. user_gt using the Hungarian Algorithm.
    Returns (TP, FP, FN).

    user_pred: DataFrame with intervals: ["pred_start", "pred_end"]
    user_gt:   DataFrame with intervals: ["gt_start",   "gt_end"  ]
    """
    traj_stops = trajectories[trajectories['labels'] != -1].copy().reset_index()
    traj_stops.sort_values(["user_id", "datetime"], inplace=True)
    traj_stops['moved'] = traj_stops.groupby(by='user_id').apply(lambda x: (x['labels'] != x['labels'].shift()).cumsum()).droplevel(0)
    det_stop_intervals = (
        traj_stops
            .groupby(["user_id", "moved"], as_index=False)
            .agg(
            start_time=("datetime", "first"),
            end_time=("datetime", "last"),
        )
    )

    traj_stops = trajectories[trajectories['labels_reference'] != -1].copy().reset_index()
    traj_stops.sort_values(["user_id", "datetime"], inplace=True)
    traj_stops['moved'] = traj_stops.groupby(by='user_id').apply(
        lambda x: (x['labels_reference'] != x['labels_reference'].shift()).cumsum()).droplevel(0)
    true_stop_intervals = (
        traj_stops
            .groupby(["user_id", "moved"], as_index=False)
            .agg(
            gt_start_time=("datetime", "first"),
            gt_end_time=("datetime", "last"),
        )
    )
    all_users = true_stop_intervals.user_id.unique()
    results = []
    for user in all_users:
        user_pred = det_stop_intervals[det_stop_intervals['user_id'] == user].copy()
        user_gt = true_stop_intervals[true_stop_intervals['user_id'] == user].copy()

        if user_pred.empty and user_gt.empty:
            # Nothing to match, no positives, no negatives => perfect in a sense
            return (0, 0, 0)
        if user_pred.empty:
            # All ground truth are missed => FN = len(user_gt)
            return (0, 0, len(user_gt))
        if user_gt.empty:
            # We predicted intervals but there's no GT => all predicted are false
            return (0, len(user_pred), 0)

            # Build cost matrix, shape = (#predicted, #ground_truth)
        n_pred = len(user_pred)
        n_gt = len(user_gt)
        cost_matrix = np.zeros((n_pred, n_gt), dtype=np.float64)

        # Compute cost = (1 - IoU) or large number if IoU < threshold
        for i in range(n_pred):
            pstart = user_pred.iloc[i]["start_time"]
            pend = user_pred.iloc[i]["end_time"]
            for j in range(n_gt):
                gstart = user_gt.iloc[j]["gt_start_time"]
                gend = user_gt.iloc[j]["gt_end_time"]
                iou = interval_iou(pstart, pend, gstart, gend)
                if iou >= iou_threshold:
                    cost_matrix[i, j] = 1.0 - iou
                else:
                    cost_matrix[i, j] = 9999.0  # "Impossible" match if below threshold

        # Solve the assignment problem: find minimal total cost
        # linear_sum_assignment => row_ind, col_ind
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Count how many matches are "valid" => cost < 9999
        # i.e. those with IoU >= iou_threshold
        valid_matches = 0
        matched_pred = set()
        matched_gt = set()
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 9999.0:
                # It's a valid match => IoU >= threshold
                valid_matches += 1
                matched_pred.add(r)
                matched_gt.add(c)

        TP = valid_matches
        FP = n_pred - TP  # predicted intervals that didn't get matched
        FN = n_gt - TP  # ground-truth intervals that didn't get matched

        precision = TP / float(TP + FP) if (TP + FP) else 0.0
        recall = TP / float(TP + FN) if (TP + FN) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
        results.append({
            "user_id": user,
            "b_TP": TP,
            "b_FP": FP,
            "b_FN": FN,
            "b_Precision": precision,
            "b_Recall": recall,
            "b_F1": f1
        })
    results = pd.DataFrame(results)
    results.set_index('user_id', inplace=True)
    return results