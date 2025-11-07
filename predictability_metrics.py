import pandas as pd
import numpy as np
from tqdm import tqdm
from math import ceil
from utils import matchfinder, _fit_func
from random import sample
from scipy.optimize import curve_fit, fsolve
from sklearn.metrics import r2_score
import concurrent.futures as cf
from Bio import pairwise2
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

tqdm.pandas()


def _real_entropy(indi, gs):
    """
    Calculates actual entropy for series of symbols
    Args:
        indi: unique identifier
        gs: series of symbols
    Returns:
        a unique identifier and entropy value
    """
    return indi, np.power(np.mean(matchfinder(gs)), -1) * np.log2(len(gs))


def _real_scalling_entropy(indi, trace, estimation_method='unc'):
    """
    Calculates actual entropy for trajectories. If trajectory has missing data, uses estimation. Uncorrelated entropy-based
    estimation is used.
    Args:
        indi: unique identifier
        trace: movement trajectory
    Returns:
        a unique identifier and actual entropy
    """
    empty_fraction = trace.isnull().sum() / trace.shape[0]
    if empty_fraction < .2: #.15
        return _real_entropy(indi, trace)
    elif empty_fraction > .9:
        raise ValueError("Fraction of missing records (q = %f) to high to estimate real entropy!" % empty_fraction)
    estimation_step = ceil((.9 - empty_fraction) / .05)
    range_to_empty = [empty_fraction + .05 * x for x in range(estimation_step)]
    scaling_features = []
    uncs = []
    real_qs = []
    if estimation_method == 'unc':
        visit_freq = trace.value_counts() / trace.shape[0]  # FOR UNCORRELATED ENTROPY-BASED ESTIMATION
        Sunc_baseline = -np.sum(visit_freq * np.log2(visit_freq))  # FOR UNCORRELATED ENTROPY-BASED ESTIMATION
    elif estimation_method == 'shuff':
        Sunc_baseline = np.power(np.mean(matchfinder(trace.sample(frac=1).reset_index(drop=True))), -1) * \
                        np.log2(len(
                            trace.sample(frac=1).reset_index(drop=True)))  # FOR SHUFFLED TRAJECTORY-BASED ESTIMATION
    for q in range_to_empty[1:]:
        trace_copy2 = trace.copy()
        points_to_remove = sample(set(trace_copy2[~trace_copy2.isnull()].index),
                                  int(round((q - empty_fraction) * len(trace_copy2))))
        trace_copy2.loc[points_to_remove] = None
        Strue = np.power(np.mean(matchfinder(trace_copy2)), -1) * np.log2(len(trace_copy2))
        trace_shuffled = trace_copy2.sample(frac=1).reset_index(drop=True)
        if estimation_method == 'unc':
            visit_freq = trace_copy2.value_counts() / trace_copy2.shape[0]  # FOR UNCORRELATED ENTROPY-BASED ESTIMATION
            Sunc = -np.sum(visit_freq * np.log2(visit_freq))  # FOR UNCORRELATED ENTROPY-BASED ESTIMATION
        elif estimation_method == 'shuff':
            Sunc = np.power(np.mean(matchfinder(trace_shuffled)), -1) * np.log2(
                len(trace_shuffled))  # FOR SHUFFLED TRAJECTORY-BASED ESTIMATION
        scaling_features.append(np.log2(Strue / Sunc))
        uncs.append(Sunc)
        real_qs.append(sum(trace_copy2.isnull()) / len(trace_copy2))
    try:
        popt, pcov = curve_fit(_fit_func, real_qs, scaling_features, maxfev=12000, p0=[0.1, 2, 0.1])
        if sum(scaling_features) == 0 and r2_score(scaling_features, [_fit_func(x, *popt) for x in real_qs]) < .9:
            a, b = np.polyfit(real_qs, scaling_features, 1)
            return indi, np.power(2, b) * Sunc_baseline
        else:
            return indi, np.power(2, _fit_func(0, *popt)) * Sunc_baseline
    except:
        a, b = np.polyfit(real_qs, scaling_features, 1)
        return indi, np.power(2, b) * Sunc_baseline


def real_entropy(trajectories):
    """
    Calculates an actual entropy for each user in TrajectoriesFrame

    Args:
        trajectories: TrajectoriesFrame class object

    Returns:
        a Series with actual entropies for each user
    """
    result_dic = {}
    with cf.ThreadPoolExecutor() as executor:
        try:
            args = [val.labels for indi, val in trajectories.groupby(level=0)]
        except KeyError:
            args = [val for indi, val in trajectories.groupby(level=0)]
        ids = [indi for indi, val in trajectories.groupby(level=0)]
        results = list(executor.map(_real_scalling_entropy, ids, args))
    for result in results:
        result_dic[result[0]] = result[1]
    return pd.Series(result_dic)


def _iterative_global_align(s1, s2):
    one = list(s1)  # prepare lists of symbols
    two = list(s2)
    cut = two  # assign currently processed sequence
    all_match = []
    to_search = []
    while True:
        best_match = \
            pairwise2.align.globalms(one, cut, 1, -1, -1, 0, penalize_end_gaps=False,
                                     one_alignment_only=True, gap_char=['-'])[
                0]
        zipped = [(x, y) for x, y in zip(best_match[0], best_match[1])]  # combinations of matched symbols
        out_of_match = [1 if x[0] != x[1] and x[0] == '-' else 0 for x in zipped]  # search for mismatched symbols
        out_diff = np.diff(out_of_match)  # find gaps
        starts = [x for x in range(len(out_diff)) if out_diff[x] == 1]  # find starts of gaps
        ends = [x for x in range(len(out_diff)) if out_diff[x] == -1]  # find ends of gaps
        lengths = [y - x for x, y in zip(starts, ends)]  # find lengths of gaps
        first_ends = [1 if x == '-' else 0 for x in best_match[0]]  # find mismatched starts and ends of sequence
        first_ends_diff = np.diff(first_ends)  # find range of mismatched starts and ends
        first_ends_starts = [x for x in range(len(first_ends_diff)) if first_ends_diff[x] == 1]
        first_ends_ends = [x for x in range(len(first_ends_diff)) if first_ends_diff[x] == -1]
        if len(first_ends_ends) > 0 and 0 not in first_ends[:first_ends_ends[0]]:
            begin = best_match[1][:first_ends_ends[0] + 1]
        else:
            begin = []
        if len(first_ends_starts) > 0 and 0 not in first_ends[first_ends_starts[-1] + 1:]:
            end = best_match[1][first_ends_starts[-1] + 1:]
        else:
            end = []
        lengths += [len(begin), len(end)]
        maxleng = np.max(lengths)  # check which mismatch is the longest
        longest = np.where(lengths == maxleng)  # take the mismatched part
        if longest[0][0] == len(lengths) - 1:  # check if its a gap in the middle or at the ends
            to_search.append(end)
        elif longest[0][0] == len(lengths) - 2:
            to_search.append(begin)
        else:
            for n in longest[0]:
                n = n
                if n == len(lengths) - 1:
                    to_search.append(end)
                elif n == len(lengths) - 2:
                    to_search.append(begin)
                else:
                    to_search.append(best_match[1][starts[n] + 1:ends[n] + 1])
        if len(to_search) == 0:  # if there are no sequences to match - stop
            if best_match[2] > 0:  # if there was positive score in the last match - add it to the list
                all_match.append(best_match[2] - 1)  # add score to the list of scores (-1 for transitions)
            break
        cut = to_search.pop(0)  # pop out the sequence for search
        cut = [x for x in cut if isinstance(x, float)]  # take only symbols
        if best_match[2] <= 0:  # if there is zero score already - stop
            break
        all_match.append(best_match[2] - 1)  # add score to the list of scores (-1 for transitions)
        if len(cut) <= 1:
            break  # if sequence for search does not consist of at least two symbols - stop
    return sum(all_match) / (len(two) - 1)


def _iterative_global_alignment(indi, S1, S2):
    S1 = S1.values
    S2 = S2.values
    return indi, _iterative_global_align(S1, S2)


def iterative_global_alignment(train_frame, test_frame):
    """
    Calculates the iterative global alignment metric proposed in the paper Smolak et al. (2022) through the application
    of the Needleman-Wunsch algorithm on sequences for pairwise matching with the iterative approach.
    Returns normalised values.

    Args:
        train_frame: TrajectoriesFrame class object with the training data
        test_frame: TrajectoriesFrame class object with the test data

    Returns:
        a Series with global alignment metric values for each user
    """
    result_dic = {}
    with cf.ThreadPoolExecutor() as executor:
        args_train = [val.labels for indi, val in train_frame.groupby(level=0)]
        args_test = [val.labels for indi, val in test_frame.groupby(level=0)]
        ids = [indi for indi, val in test_frame.groupby(level=0)]
        results = list(tqdm(executor.map(_iterative_global_alignment, ids, args_train, args_test), total=len(ids)))
    for result in results:
        result_dic[result[0]] = result[1]
    return pd.Series(np.fromiter(result_dic.values(), dtype=float), index=np.fromiter(result_dic.keys(), dtype=int))


def fano_inequality(distinct_locations, entropy):
    """
    Implementation of the Fano's inequality. Algorithm solves it and returns the solution.
    :param distinct_locations:
    :param entropy:
    :return:
    """
    func = lambda x: (-(x * np.log2(x) + (1 - x) * np.log2(1 - x)) + (1 - x) * np.log2(
        distinct_locations - 1)) - entropy
    return fsolve(func, .9999)[0]


def num_of_distinct_locations(trajectories_frame, column_name = 'labels'):
    """
    Returns a number of distinct location in the trajectory. First looks for 'labels' column.

    Args:
        trajectories_frame: TrajectoriesFrame class object

    Returns:
        a Series with the number of unique locations for each user
    """
    if isinstance(trajectories_frame, pd.DataFrame):
        return trajectories_frame.groupby(level=0).apply(lambda x: len(pd.unique(x[column_name])))
    else:
        return trajectories_frame.groupby(level=0).apply(lambda x: pd.unique(x).shape[0])


def real_predictability(trajectories_frame):
    """
    Calculates actual entropy and predictability.

    Args:
        trajectories_frame: TrajectoriesFrame class object

    Returns:
        a Series with actual entropy and predictability for each user
    """
    distinct_locations = num_of_distinct_locations(trajectories_frame)
    real_ent = real_entropy(trajectories_frame)
    merged = pd.DataFrame([distinct_locations, real_ent], index=['locations', 'entropy'])
    return merged.apply(lambda x: fano_inequality(x['locations'], x['entropy'])), real_ent
