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
