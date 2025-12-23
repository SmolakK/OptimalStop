"""
Example pipeline for parameter optimisation and evaluation using OptimalStop.
"""

import pandas as pd
from geopandas import gpd
from optimal_stop import OptimalStop
from evaluation_metrics import (
    overlap_fast,
    missed_fast,
    overdetected_fast,
    undersegmentation_fast,
    oversegmentation_fast,
)

# ------------------------------------------------------------------
# 1. Experimental configuration
# ------------------------------------------------------------------

RUN_PARAMETERS = {
    "detector": "lachesis",            # alternatives: "stdbscan", "infostop"
    "temporal_resampling": "30min",    # or None
    "param_space": {
        "stop_distance": (2, 20),    # meters
        "stop_time": (3 * 60, 15 * 60),# seconds
        "eps": (2, 20),             # meters
        "min_samples": (1, 3),
    },
}

# Detector-specific parameters:
# - Infostop: r1_level, r2_level, min_staying_time
# - ST-DBSCAN: eps, eps2, eps3, min_samples


# ------------------------------------------------------------------
# 2. Load data
# ------------------------------------------------------------------

# Trajectory data
df = pd.read_csv("synthetic_ground_truth/random_city.csv")
df.drop('location',axis=1,inplace=True) # For the sake of this example
df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['lon'],df['lat']),crs=3857)

# Ground truth stop labels (optional)
ref_df = pd.read_csv("synthetic_ground_truth/random_city.csv")
ref_df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(ref_df['lon'],ref_df['lat']))


# ------------------------------------------------------------------
# 3. Run parameter optimisation
# ------------------------------------------------------------------

ostop = OptimalStop(
    df=df.copy(),
    detector=RUN_PARAMETERS["detector"],
    temporal_resampling=RUN_PARAMETERS["temporal_resampling"],
    param_space=RUN_PARAMETERS["param_space"],
)

# Bayesian optimisation using Optuna
ostop.fit_predict(n_trials=200)

# Select the optimal configuration
picks = ostop.select_best().reset_index()


# ------------------------------------------------------------------
# 4. Evaluation against ground truth
# ------------------------------------------------------------------

# Align predictions with reference labels
merged = pd.merge(
    ref_df,
    picks,
    on=["user_id", "datetime"],
    how="inner",
)

merged = (
    merged[["user_id", "datetime", "labels_x", "labels_y"]]
    .rename(
        columns={
            "labels_x": "labels_reference",
            "labels_y": "labels_predicted",
        }
    )
)

# ------------------------------------------------------------------
# 5. Compute evaluation metrics (optional)
# ------------------------------------------------------------------

scores = overlap_fast(merged)
missed = missed_fast(merged)
overdetected = overdetected_fast(merged)
underseg = undersegmentation_fast(merged)
overseg = oversegmentation_fast(merged)

# Harmonic mean of predicted and reference IoU
iou_f = (
    2 * scores["IOU_PRED"] * scores["IOU_GT"]
    / (scores["IOU_PRED"] + scores["IOU_GT"])
)

# Aggregate all metrics
results = pd.concat(
    [
        scores,
        missed,
        overdetected,
        underseg,
        overseg,
        iou_f.rename("IoU_F"),
    ],
    axis=1,
)
