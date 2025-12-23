# OptimalStop: Model-agnostic stop detection algorithms tuning

**OptimalStop** is a model-agnostic framework for optimising stop-detection algorithms in mobility data.  
It automatically tunes stop-detection parameters using **multi-objective Bayesian optimisation**.

The core idea is to balance:

- **Predictability** of the resulting symbolic trajectory  
- **Complexity** (information content / entropy)

The framework is designed to work with high-resolution movement trajectories collected via GPS or other terrestrial positioning systems. It supports both **human and animal mobility data**, handles **irregular sampling**, and can be applied **with or without ground-truth labels**.

---

## Key Features

- **Model-agnostic optimisation**  
  Works with built-in stop detectors as well as user-defined algorithms through a unified interface.

- **Multi-objective Bayesian optimisation**  
  Efficient exploration of parameter spaces using Optuna, jointly optimising predictability and entropy.

- **Degeneracy-aware optimisation**  
  Trivial solutions (e.g. single-cluster or dominant-label outputs) are explicitly penalised.

- **Individual-level parameter tuning**  
  Parameters are optimised separately for each trajectory.

- **Support for irregular sampling**  
  Optional temporal resampling into fixed time bins to reduce bias from heterogeneous sampling rates.

- **Ground-truth–free by design**  
  No reference labels are required for optimisation; evaluation against ground truth is optional.

- **Evaluation utilities**  
  Includes overlap (IoU-based) metrics, missed and over-detected stops, and under- and over-segmentation diagnostics.

## Supported Stop Detection Algorithms

- **Project Lachesis**
Commonly used threshold-based stop detection algorithm combined with DBSCAN for refinment. Adopted from HuMobi library. (Parameters: `stop_distance`, `stop_time`, `eps`, `min_samples`).
- **ST-DBSCAN**
Spatio-temporal DBSCAN + spatial refinement, implemented from: `Birant, D., & Kut, A. (2007). ST-DBSCAN: An algorithm for clustering spatial–temporal data. Data & knowledge engineering, 60(1), 208-221.` (Parameters: `eps`, `eps2`, `eps3`, `min_samples`).
- **Infostop**
Information-theoretic stop detection implemented from `Aslak, U., & Alessandretti, L. (2020). Infostop: scalable stop-location detection in multi-user mobility data. arXiv preprint arXiv:2003.14370.`. (Parameters: `r1_level`, `r2_level`, `min_staying_time`).

- **Custom detectors**
Your own detector, as long as it implements `.fit_predict(df, params) -> DataFrame with 'labels' column`.

## Installation
Currently intended for research use (editable install recommended):
git clone https://github.com/SmolakK/OptimalStop
cd OptimalStop
pip install -e .

Dependencies include:
- pandas, geopandas
- scikit-learn
- optuna
- numba
- pyproj

## Data Requirements
Input trajectories must contain:
`user_id | datetime | lat | lon`

- datetime must be convertible to datetime64
- CRS defaults to EPSG: 4326 (unless specified in geopandas DataFrame)

## Usage example
The file `usage_example.py` contains the full usage pipeline.
Using OptimalStop is as simple as:
- Loading your data to a DataFrame or GeoDataFrame
- Initilising OptimalStop object:
  `ostop = OptimalStop(
    df=df.copy(),
    detector=RUN_PARAMETERS["detector"],
    temporal_resampling=RUN_PARAMETERS["temporal_resampling"],
    param_space=RUN_PARAMETERS["param_space"],
  )`
- Running optimisation: `ostop.fit_predict(n_trials=200)`
- Optionally picking the "best" solutions: `ostop.select_best()`

## License
MIT License.
Free for academic and industrial use.

## Code for publication

All the codes for the publication "OptimalStop: Predictability–-Complexity Trade-off for Optimising Stop Detection in Human Mobility Data" can be found in `paper_codes` subfolder. It includes 'experiments_*' which were used to calculate various stop-detection setups and measure their accuracy, and 'results_analysis', which were used to analyse outputs.
