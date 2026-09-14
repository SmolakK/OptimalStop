"""
Example pipeline for parameter optimisation and evaluation using OptimalStop.
"""

import pandas as pd
import geopandas as gpd
from optimal_stop import OptimalStop
from evaluation_metrics import (
    overlap_fast,
    missed_fast,
    overdetected_fast,
    undersegmentation_fast,
    oversegmentation_fast,
)


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


raw = pd.read_csv("synthetic_ground_truth/random_city.csv", index_col=0)
raw["datetime"] = pd.to_datetime(raw["datetime"])

# Ground truth stop labels (optional): 'location' holds the true stop id, -1 = moving
ref_df = raw[["user_id", "datetime", "location"]].rename(columns={"location": "labels_reference"})

# Trajectory data (ground truth removed; synthetic coordinates are in metres, EPSG:3857)
df = raw.drop(columns="location")
df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=3857)


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


# Align predictions ('labels') with reference labels ('labels_reference')
merged = pd.merge(
    ref_df,
    picks[["user_id", "datetime", "labels"]],
    on=["user_id", "datetime"],
    how="inner",
)


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
print(results.describe())
