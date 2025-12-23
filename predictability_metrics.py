import pandas as pd
import numpy as np
from tqdm import tqdm
from math import ceil
from utils import matchfinder, _fit_func
from random import sample
from scipy.optimize import curve_fit, fsolve
from sklearn.metrics import r2_score
import concurrent.futures as cf
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

tqdm.pandas()


def _real_entropy(indi, gs):
    """
    Computes the Lempel–Ziv-based entropy estimator for a sequence of symbols.
    Args:
        indi: unique identifier
        gs: series of symbols
    Returns:
        a unique identifier and entropy value
    """
    return indi, np.power(np.mean(matchfinder(gs)), -1) * np.log2(len(gs))


def _real_scalling_entropy(indi, trace, estimation_method='unc'):
    """
    Estimates the real (Lempel–Ziv) entropy of a trajectory with missing data.

    If the fraction of missing observations is small, entropy is computed directly.
    For moderate missingness, entropy is extrapolated to the zero-missing limit
    using a scaling relationship between true entropy and an uncorrelated baseline
    (see Song et al., Science, 2010).

    Trajectories with excessive missingness are discarded.
    Args:
        indi: unique identifier
        trace: movement trajectory
        estimation_method: used estimation method when data are missing (default = 'unc', uncorrelated baseline)
    Returns:
        a unique identifier and actual entropy
    """
    empty_fraction = trace.isnull().sum() / trace.shape[0]
    if empty_fraction < .2: # Below 20% missing data: direct entropy estimation is reliable
        return _real_entropy(indi, trace)
    elif empty_fraction > .9: # Below 20% missing data: direct entropy estimation is reliable
        raise ValueError("Fraction of missing records (q = %f) to high to estimate real entropy!" % empty_fraction)
    estimation_step = ceil((.9 - empty_fraction) / .05)
    range_to_empty = [empty_fraction + .05 * x for x in range(estimation_step)] # Missing fractions are increased in 5% steps for scaling extrapolation
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
        # If nonlinear scaling fails or fit quality is poor,
        # fall back to linear extrapolation in log-space
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
