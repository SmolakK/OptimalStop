import numpy as np
import pandas as pd
from tqdm import tqdm
from predictability_metrics import real_entropy, real_predictability, num_of_distinct_locations, fano_inequality
import networkx as nx
from collections import Counter
from igraph import Graph


def random_entropy(trajectories, column_name = 'labels'):
    """
    Calculates random entropy for each user in TrajectoriesFrame
    Args:
        trajectories: TrajectoriesFrame class object
    Returns:
        a Series with random entropies for each user
    """
    return trajectories.groupby(level=0).progress_apply(lambda x: np.log2(len(pd.unique(x[column_name]))))


def random_predictability(trajectories_frame, column_name):
    """
    Calculates random entropy and predictability.

    Args:
        trajectories_frame: TrajectoriesFrame class object

    Returns:
        a Series with random entropy and predictability for each user
    """
    distinct_locations = num_of_distinct_locations(trajectories_frame, column_name)
    rand_ent = random_entropy(trajectories_frame, column_name)
    merged = pd.DataFrame([distinct_locations, rand_ent], index=['locations', 'entropy'])
    return merged.progress_apply(lambda x: fano_inequality(x['locations'], x['entropy'])), rand_ent


def clusters_gini(trajectories, column_name = 'labels'):
    entropies = {}
    for user_id, group in tqdm(trajectories.groupby(level=0)):
        group = group.sort_values('datetime')
        group['next_time'] = group['datetime'].shift(-1)
        group = group[group[column_name] != -1]
        group['moved'] = (group[column_name] != group[column_name].shift()).cumsum()
        visits = group.groupby(['moved', column_name]).agg(
            start_time=('datetime', 'first'),
            end_time=('next_time', 'last')
        ).reset_index()
        visits['time_spent'] = (visits['end_time'] - visits['start_time']).dt.total_seconds()
        dwell_times = visits.time_spent.values
        sorted_times = np.sort(dwell_times)
        cumulative_sum = np.cumsum(sorted_times)
        total_sum = np.sum(sorted_times)
        n = len(dwell_times)
        if total_sum == 0:
            return 0.0

        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_times))) / (n * total_sum) - (n + 1) / n
        entropies[user_id] = gini
    return pd.Series(entropies)


def clusters_time_entropy(trajectories, column_name ='labels', normalize = True, filtr = 1e-15, scaling_type = 'Total'):
    entropies = {}
    for user_id, group in tqdm(trajectories.groupby(level=0)):
        total_hours = (group.datetime.max() - group.datetime.min()).total_seconds() / (60 * 60)
        group = group.sort_values('datetime')
        group['next_time'] = group['datetime'].shift(-1)
        group = group[group[column_name] != -1]
        if group.empty:
            entropies[user_id] = 0
            continue
        group['moved'] = (group[column_name] != group[column_name].shift()).cumsum()
        visits = group.groupby(['moved', column_name]).agg(
            start_time=('datetime', 'first'),
            end_time=('next_time', 'last')
        ).reset_index()
        visits['time_spent'] = (visits['end_time'] - visits['start_time']).dt.total_seconds()/(60*60)
        if scaling_type == "Total":
            time_ratio = visits.groupby(column_name).apply(lambda x: x.time_spent.sum())/total_hours
        elif scaling_type == "Visits":
            time_ratio = visits.groupby(column_name).apply(
                lambda x: x.time_spent.sum()) / visits.time_spent.sum()
        time_ratio = time_ratio.values
        time_ratio = time_ratio[time_ratio > filtr]

        # Calculate Shannon entropy
        entropy = -np.sum(time_ratio * np.log2(time_ratio))
        if normalize:
            entropy /= np.log2(total_hours)
        entropies[user_id] = entropy
    return pd.Series(entropies)


def clusters_entropy(trajectories, total_points, column_name ='labels', filtr = 1e-15, normalize = True,
                     scaling_type = 'Points'):
    entropies = {}
    for user_id, group in tqdm(trajectories.groupby(level=0)):
        group = group.sort_values('datetime').copy()
        user_points = total_points.loc[user_id]
        group_de1 = group[group[column_name] != -1]
        group_de1['moved'] = (group_de1[column_name] != group_de1[column_name].shift()).cumsum()
        visits = group_de1.groupby(['moved',column_name]).size().reset_index()
        time_ratio = visits.groupby(column_name).sum().iloc[:,-1]
        time_ratio /= time_ratio.sum()
        time_ratio = time_ratio.values
        time_ratio = time_ratio[time_ratio > filtr]
        entropy = -np.sum(time_ratio * np.log2(time_ratio))
        if normalize:
            if scaling_type == 'Points':
                entropy /= np.log2(user_points)
            elif scaling_type == 'Stops':
                entropy /= np.log2(group_de1[column_name].shape[0])
        entropies[user_id] = entropy
    return pd.Series(entropies)


def transition_entropy(trajectories, total_points, column_name ='labels', travels = False, filr = 0.01):
    entropies = {}
    for user_id, group in tqdm(trajectories.groupby(level=0)):
        group = group.sort_values('datetime').copy()
        if travels:
            group_de1 = group
        else:
            group_de1 = group[group[column_name] != -1]
        user_points = total_points.loc[user_id]
        labels = group_de1[column_name].values
        # Count occurrences of each cluster
        unique_labels = np.unique(labels)
        n_states = len(unique_labels)

        transition_matrix = np.zeros((n_states, n_states))

        state_to_index = {state: idx for idx, state in enumerate(unique_labels)}
        for i in range(len(labels) - 1):
            from_state = state_to_index[labels[i]]
            to_state = state_to_index[labels[i+1]]
            transition_matrix[from_state, to_state] += 1

        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums, out=np.zeros_like(transition_matrix),
                                      where=row_sums != 0)
        transition_matrix[transition_matrix < 0.01] = 0
        transition_entropy = 0
        for i in range(n_states):
            for j in range(n_states):
                if transition_matrix[i, j] > 1e-15:
                    transition_entropy -= transition_matrix[i, j] * np.log2(transition_matrix[i, j])
        transition_entropy /= (np.log2(user_points * (user_points - 1)))
        entropies[user_id] = transition_entropy
    return pd.Series(entropies)


def shuffled_entropy(trajectories):
    res = []
    for n in range(5):
        res.append(real_predictability(trajectories.groupby(level=0).apply(lambda x: x.sample(frac=1)))[0])
    return pd.concat(res,axis=1).mean(axis=1)


def transitions(trajectories, length = 2):
    entropies = {}
    for user_id, group in trajectories.groupby(level=0):
        labels = group.labels
        path_counts = Counter()
        for i in range(len(labels) - length + 1):
            # Extract a subpath of given length
            subpath = tuple(labels[i:i + length])
            path_counts[subpath] += 1

        unique_paths = len(path_counts.keys())
        total_count = sum(path_counts.values())

        if total_count > 0:
            # Convert frequencies to probabilities
            path_probabilities = [count / total_count for count in path_counts.values()]
            # Compute entropy
            path_entropy = -sum(p * np.log2(p) for p in path_probabilities if p > 0)
        # path_entropy /= np.log2(unique_paths)
        entropies[user_id] = path_entropy
    return pd.DataFrame.from_dict(entropies,orient='index')

def graph_complexity(trajectories):
    in_degrees_users = {}
    out_degress_users = {}
    C_users = {}
    Q_users = {}
    in_entropy_users = {}
    out_entropy_users = {}
    motif_entropy_users = {}
    path_entropy_users = {}
    for user_id, group in tqdm(trajectories.groupby(level=0)):
        labels = group.labels
        labels = labels[labels != labels.shift()]
        edges = [(labels.iloc[x], labels.iloc[x + 1]) for x in range(len(labels) - 1)]

        # Build a directed graph
        G = nx.DiGraph()
        G.add_edges_from(edges)

        out_degrees = dict(G.out_degree())  # node -> out-degree
        in_degrees = dict(G.in_degree())  # node -> in-degree

        # Compute basic statistics about degree distribution
        out_degree_vals = list(out_degrees.values())
        in_degree_vals = list(in_degrees.values())

        average_out_degree = sum(out_degree_vals) / len(out_degree_vals) if out_degree_vals else 0
        average_in_degree = sum(in_degree_vals) / len(in_degree_vals) if in_degree_vals else 0

        C = nx.average_clustering(G.to_undirected())

        communities = nx.community.greedy_modularity_communities(G.to_undirected())
        Q = nx.algorithms.community.quality.modularity(G.to_undirected(), communities)

        if nx.is_strongly_connected(G):
            avg_sp = nx.average_shortest_path_length(G, weight=None)

        degree_counts = Counter(out_degree_vals)
        total_nodes = G.number_of_nodes()
        if total_nodes > 0:
            p = [count / total_nodes for count in degree_counts.values()]
            out_degree_entropy = -sum(pi * np.log2(pi) for pi in p if pi > 0)

        degree_counts = Counter(in_degree_vals)
        if total_nodes > 0:
            p = [count / total_nodes for count in degree_counts.values()]
            in_degree_entropy = -sum(pi * np.log2(pi) for pi in p if pi > 0)

        edges = list(G.edges())
        ig = Graph(directed=True)
        ig.add_vertices(len(G.nodes()))
        mapping = {node: i for i, node in enumerate(G.nodes())}
        ig.add_edges((mapping[u], mapping[v]) for u, v in edges)
        triad_counts = np.array(ig.motifs_randesu(size=3))
        triad_counts = triad_counts[~np.isnan(triad_counts)]
        total = sum(triad_counts)
        if total > 0:
            p = [count / total for count in triad_counts]
            motif_entropy = -sum(prob * np.log2(prob) for prob in p if prob > 0)

        subpath_lengths = [2]

        path_counts = Counter()

        for length in subpath_lengths:
            for i in range(len(labels) - length + 1):
                # Extract a subpath of given length
                subpath = tuple(labels[i:i + length])
                path_counts[subpath] += 1

        total_count = sum(path_counts.values())

        if total_count > 0:
            # Convert frequencies to probabilities
            path_probabilities = [count / total_count for count in path_counts.values()]
            # Compute entropy
            path_entropy = -sum(p * np.log2(p) for p in path_probabilities if p > 0)

        path_entropy_users[user_id] = path_entropy
        in_degrees_users[user_id] = average_in_degree
        out_degress_users[user_id] = average_out_degree
        C_users[user_id] = C
        Q_users[user_id] = Q
        in_entropy_users[user_id] = in_degree_entropy
        out_entropy_users[user_id] = out_degree_entropy
        motif_entropy_users[user_id] = motif_entropy
    return [pd.DataFrame.from_dict(x,orient='index') for x in
            [in_degrees_users,out_degress_users,C_users,Q_users,in_entropy_users,out_entropy_users,motif_entropy_users,
             path_entropy_users]]

from scipy.special import psi  # digamma function


def sg_entropy(trajectories, column_name = 'labels'):
    """
    Schürmann–Grassberger entropy estimator for discrete distributions.

    Parameters:
    - counts: array-like, counts of observations (e.g., time in locations)

    Returns:
    - estimated entropy in bits
    """
    entropies = {}
    for user_id, group in tqdm(trajectories.groupby(level=0)):
        group_de1 = group[group[column_name] != -1]
        counts = np.unique(group_de1.labels,return_counts=True)[1]
        counts = np.asarray(counts, dtype=float)
        n = np.sum(counts)
        k = np.count_nonzero(counts)

        if n <= 1 or k <= 1:
            entropies[user_id] = entropy

        entropy = (
            psi(n + 1)
            - (1 / n) * np.sum(counts * psi(counts + 1))
            + np.log2(np.e)  # convert from nats to bits
        )
        entropies[user_id] = entropy
    return pd.DataFrame.from_dict(entropies,orient='index')