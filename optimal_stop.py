import optuna
import pandas as pd
import geopandas as gpd
import warnings
from utils import stays_to_slots_longest_fast
from tqdm import tqdm
from complexity_metrics import real_predictability
from pyproj import Transformer
from stop_detection import ClusteringAggregator
from st_dbscan import ST_DBSCAN
from sklearn.cluster import DBSCAN
from stop_detection import infostop
from optuna.exceptions import TrialPruned


class OptimalStop:
    """
    Optimiser for stop detection algorithms using multi-objective Optuna search. Allows user to optimise stop detection
    parameters picking solution out of Pareto Front or using "ideal" solution.

    Supports:
    - built-in algorithms (Infostop, ST-DBSCAN, StopGo, Project Lachesis-like)
    - custom detector objects implementing .fit_predict(df) -> labels (Series indexed by df)
    """

    REQUIRED_COLUMNS = {"user_id", "datetime", "lat", "lon"}
    BUILTIN_DETECTORS = {"lachesis", "stdbscan", "infostop"}
    DETECTOR_REQUIRED_PARAMS = {
        "lachesis": {"stop_distance", "stop_time", "eps", "min_samples"},
        "stdbscan": {"eps", "eps2", "eps3", "min_samples"},
        "infostop": {"r1_level", "r2_level", "min_staying_time"},
    }
    DEFAULT_PARAM_SPACES = {
        "lachesis": {
            "stop_distance": lambda t: t.suggest_float("stop_distance", 10, 300, step=5),
            "stop_time": lambda t: t.suggest_int("stop_time", 60, 60 * 20, step=300),
            "eps": lambda t: t.suggest_float("eps", 10, 200, step=5),
            "min_samples": lambda t: t.suggest_int("min_samples", 1, 10),
        },
        "stdbscan": {
            "eps": lambda t: t.suggest_float("eps", 5, 100, step=5),
            "eps2": lambda t: t.suggest_float("eps2", 30, 600, step=30),  # temporal threshold
            "eps3": lambda t: t.suggest_float("eps3", 5, 200, step=5),  # refinement eps
            "min_samples": lambda t: t.suggest_int("min_samples", 1, 10),
        },
        "infostop": {
            "r1_level": lambda t: t.suggest_float("r1_level", 10, 200, step=5),
            "r2_level": lambda t: t.suggest_float("r2_level", 10, 300, step=5),
            "min_staying_time": lambda t: t.suggest_int("min_staying_time", 60, 1800, step=60),
        },
    }

    def __init__(
            self,
            df,
            detector,
            param_space=None,
            temporal_resampling=None,
            directions=("maximize", "maximize"),
            n_jobs=1
    ):
        """
        Args:
            df (DataFrame): Input mobility data (must include user_id, datetime, lon, lat)
            detector: Stop detection algorithm (built-in or custom)
                      MUST implement:
                          .set_params(**kwargs)
                          .fit_predict(df) -> labels
            temporal_resampling: If you want to use next time-bin approach, you might set this to pandas-like interval
            (e.g. 15min, 1H).
            param_space (dict): keys = param names, values = optuna search definitions
            directions (tuple): directions for optimising each objective
            n_jobs (int): parallel jobs for optuna
        """
        self.detector = detector
        self.temporal_resampling = temporal_resampling
        self.param_space = param_space
        self.directions = directions
        self.n_jobs = n_jobs

        self.df = self._validate_and_prepare_df(df)

        # If temporal resampling is provided → convert stays to slots
        if self.temporal_resampling is not None:
            self.perform_resampling()

        if detector is None:
            detector = "lachesis"

        if isinstance(detector, str):
            detector = detector.lower()
            if detector not in self.BUILTIN_DETECTORS:
                raise ValueError(
                    f"Unknown detector '{detector}'. Supported: {self.BUILTIN_DETECTORS}"
                )
            self.detector_name = detector
        else:
            # custom detector class
            self.detector_name = "custom"

        self.detector = detector

        self._validate_param_space()

        self.studies = []
        self.pareto = None
        self.picks = {}
        self.best_pick_data = []

    def perform_resampling(self):
        SLOT = self.temporal_resampling
        slot_ns = pd.Timedelta(SLOT).value  # 900 s  → 9.0e+11 ns
        to_conca = {}
        df_grouped = self.df.groupby(level=0, sort=False)
        for uid, g in tqdm(df_grouped, total=len(df_grouped), desc='Resampling trajectories...'):
            res = stays_to_slots_longest_fast(g, slot_ns=slot_ns)
            if res is not None:
                to_conca[uid] = res
        self.df = pd.concat(to_conca).droplevel(1)

    def _validate_and_prepare_df(self, df):
        """
        Validates structure of the input DataFrame.
        Ensures required columns, datetime format, GeoDataFrame, CRS, and indexing by user_id.
        Returns a cleaned GeoDataFrame.
        """

        # -------------------------
        # 1. Ensure DataFrame input
        # -------------------------
        if not isinstance(df, (pd.DataFrame, gpd.GeoDataFrame)):
            raise TypeError(
                "df must be a pandas DataFrame or GeoDataFrame."
            )

        df = df.copy()

        # ----------------------------------
        # 2. Check required columns (loose check: user_id may be index)
        # ----------------------------------
        actual_cols = set(df.columns)

        missing_cols = self.REQUIRED_COLUMNS - actual_cols
        # Allow user_id to be in index
        if "user_id" in missing_cols and df.index.name == "user_id":
            missing_cols.remove("user_id")

        if missing_cols:
            raise ValueError(
                f"Input DataFrame is missing required columns: {missing_cols}. "
                f"Required: {self.REQUIRED_COLUMNS}"
            )

        # -------------------------------------------------
        # 3. Ensure datetime column is datetime-like
        # -------------------------------------------------
        if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
            try:
                df["datetime"] = pd.to_datetime(df["datetime"])
            except Exception:
                raise ValueError(
                    "Column 'datetime' must be convertible to datetime."
                )

        # -------------------------------------------------
        # 4. Ensure user_id is index or move it to index
        # -------------------------------------------------
        if "user_id" in df.columns:
            df = df.set_index("user_id")
        else:
            # Already in index with correct name
            if df.index.name != "user_id":
                raise ValueError(
                    "DataFrame index must be named 'user_id' if there is no 'user_id' column."
                )

        # -------------------------------------------------
        # 5. Ensure GeoDataFrame with correct CRS
        # -------------------------------------------------
        if not isinstance(df, gpd.GeoDataFrame):
            df = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df["lon"], df["lat"]),
                crs="EPSG:4326"
            )
            warnings.warn("CRS set to 4326 by default. If it is not the correct one, pass geopandas with crs defined.")
        else:
            # If geometry exists but CRS is missing → assume 4326
            if df.crs is None:
                df = df.set_crs("EPSG:4326", inplace=False)
            # If CRS is not WGS84, reproject
            elif df.crs.to_epsg() != 4326:
                pass

        # -------------------------------------------------
        # 6. Column ordering (standardised)
        # -------------------------------------------------
        cols_order = ["datetime", "lon", "lat"]

        df = df[cols_order + ["geometry"]]

        return df

    def _run_lachesis(self, df, params):
        df = df.copy()
        df = df.reset_index()
        df = df.sort_values(['user_id', 'datetime'])

        # Ensure crs=3857 as required
        if df.crs.to_epsg() != 3857:
            transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
            df[['lon', 'lat']] = df.apply(
                lambda row: transformer.transform(row['lon'], row['lat']), axis=1,
                result_type='expand')

        clust_agg = ClusteringAggregator(
            DBSCAN,
            stop_distance=params["stop_distance"],
            stop_time=params["stop_time"],
            eps=params["eps"],
            min_samples=params["min_samples"]
        )

        aggregated_df = clust_agg.aggregate(df)
        return aggregated_df

    def _run_stdbscan(self, df, params):
        """
        Runs ST-DBSCAN + optional DBSCAN refinement for each user.

        Required params:
            eps   — spatial threshold for SpatioTemporal DBSCAN
            eps2  — temporal threshold
            eps3  — refinement DBSCAN eps
            min_samples — minimum samples for ST-DBSCAN
        """
        # Required for safety when called repeatedly inside Optuna
        df = df.copy()

        # Ensure crs=3857 as required
        if df.crs.to_epsg() != 3857:
            transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
            df[['lon', 'lat']] = df.apply(
                lambda row: transformer.transform(row['lon'], row['lat']), axis=1,
                result_type='expand')

        # Prepare UNIX timestamp
        df['unix'] = (df['datetime'].dt.tz_localize(None) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

        aggregated_df = {}

        st_dbscan = ST_DBSCAN(eps1=params['eps'], eps2=params['eps2'], min_samples=params['min_samples'])
        dbscan = DBSCAN(eps=params['eps3'], min_samples=1)

        for uid, udata in df.groupby("user_id"):
            udata = udata.copy()
            st_labels = st_dbscan.fit(udata[["unix", "lat", "lon"]]).labels
            udata["label"] = st_labels
            clustered_mask = udata['label'] != -1
            if clustered_mask.sum() > params['min_samples']:
                dj_udata = udata[clustered_mask]
                dj_lbls = dbscan.fit(dj_udata[['lat', 'lon']]).labels_
                udata.loc[clustered_mask, 'label'] = dj_lbls
            aggregated_df[uid] = udata

        aggregated_df = pd.concat(aggregated_df)
        return aggregated_df

    def _run_infostop(self, df, params):
        """
        Runs Infostop stop detection and merges results with original df.

        Required params:
            r1_level
            r2_level
            min_staying_time
        """
        df = df.copy()

        # Infostop works in EPSG:4326
        if df.crs is None or df.crs.to_epsg() != 4326:
            df = df.to_crs(4326)

        # Run Infostop
        infostop_df = infostop(
            df,
            r1=params["r1_level"],
            r2=params["r2_level"],
            min_staying_time=params["min_staying_time"]
        )
        aggregated_df = pd.concat([infostop_df, df.set_index('user_id').add_suffix('_2')], axis=1)
        return aggregated_df

    def _run_detector(self, df, params):
        if self.detector_name == "lachesis":
            return self._run_lachesis(df, params)
        if self.detector_name == "stdbscan":
            return self._run_stdbscan(df, params)
        if self.detector_name == "infostop":
            return self._run_infostop(df, params)
        if self.detector_name == "custom":
            return self.detector.fit_predict(df, params)

    def _validate_param_space(self):
        """Checks whether the provided parameter space matches the detector's required param names."""

        if self.detector_name == "custom":
            # Custom detector must implement its own parameter handling
            return

        # If no param_space provided → use defaults (for built-in detectors only)
        if self.param_space is None:
            if self.detector_name in self.DEFAULT_PARAM_SPACES:
                self.param_space = self.DEFAULT_PARAM_SPACES[self.detector_name]
            else:
                raise ValueError(
                    "param_space must be provided for custom detectors "
                    "because no default search space exists."
                )
        else:
            self.param_space = self._normalize_param_space(self.param_space)

        required = self.DETECTOR_REQUIRED_PARAMS[self.detector_name]
        provided = set(self.param_space.keys())

        missing = required - provided
        if missing:
            raise ValueError(
                f"Detector '{self.detector_name}' requires parameters {required} "
                f"but param_space is missing: {missing}"
            )

    def _normalize_param_space(self, param_space):
        """
        Converts user-provided parameter definitions into callable Optuna suggest_* functions.
        Supports:
        - already-valid callables
        - dict definitions: {"type": ..., bounds...}
        - tuple definitions: (low, high)
        """

        normalized = {}

        for name, definition in param_space.items():

            # Case A: user provided a callable → use directly
            if callable(definition):
                normalized[name] = definition
                continue

            # Case B: tuple → convert to float/int range automatically
            if isinstance(definition, tuple) and len(definition) == 2:
                low, high = definition
                if isinstance(low, int) and isinstance(high, int):
                    normalized[name] = (
                        lambda trial, n=name, l=low, h=high:
                        trial.suggest_int(n, l, h)
                    )
                else:
                    normalized[name] = (
                        lambda trial, n=name, l=low, h=high:
                        trial.suggest_float(n, l, h)
                    )
                continue

            # Case C: dict definition
            if isinstance(definition, dict):
                ptype = definition.get("type")

                if ptype == "int":
                    normalized[name] = (
                        lambda trial, n=name, d=definition:
                        trial.suggest_int(n, d["low"], d["high"], step=d.get("step"))
                    )

                elif ptype == "float":
                    normalized[name] = (
                        lambda trial, n=name, d=definition:
                        trial.suggest_float(n, d["low"], d["high"], step=d.get("step"))
                    )

                elif ptype == "categorical":
                    normalized[name] = (
                        lambda trial, n=name, d=definition:
                        trial.suggest_categorical(n, d["choices"])
                    )

                else:
                    raise ValueError(f"Unknown parameter type for '{name}': {ptype}")

                continue

            # Unsupported format
            raise ValueError(
                f"Unsupported parameter definition for '{name}': {definition}"
            )

        return normalized

    # -----------------------------------------------------------
    # PARAMETER SUGGESTION HELPERS
    # -----------------------------------------------------------
    def _suggest_params(self, trial):
        params = {}
        for name, spec in self.param_space.items():
            if spec["type"] == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"])
            elif spec["type"] == "float":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"])
            elif spec["type"] == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
            else:
                raise ValueError(f"Unknown parameter type for {name}")
        return params

    # -----------------------------------------------------------
    # OPTUNA OBJECTIVE
    # -----------------------------------------------------------
    def _objective(self, trial, df_u):

        # sample parameters from Optuna search space
        params = {
            name: suggest_fn(trial)
            for name, suggest_fn in self.param_space.items()
        }

        clustered_df = self._run_detector(df_u, params)

        # Degeneracy checks
        n_labels = clustered_df["labels"].nunique(dropna=True)
        n_points = len(clustered_df)

        # % of points in the dominant label (including noise collapsed cases)
        dom_frac = (
            clustered_df["labels"].value_counts(normalize=True, dropna=False).iloc[0]
            if n_points > 0 else 1.0
        )

        # If it’s basically one blob or almost everything got the same label → penalize
        if n_labels < 2 or dom_frac >= 0.99:
            return (1e-9, 1e-9)

        pred_h, real_h = real_predictability(clustered_df)

        if isinstance(pred_h, pd.Series):
            pred_h = float(pred_h.iloc[0])
        if isinstance(real_h, pd.Series):
            real_h = float(real_h.iloc[0])

        return pred_h, real_h

    # -----------------------------------------------------------
    # EXTERNAL START
    # -----------------------------------------------------------
    def fit_predict(self, n_trials=100):
        results = {}
        for uid in self.df.index.unique():
            print(f"--- Optimizing user {uid} ---")
            df_u = self.df.loc[[uid]]  # user subset
            pareto_front, study = self._optimize_single_user(df_u, uid, n_trials)
            self.studies.append(study)
            results[uid] = {"pareto": pareto_front}
        self.pareto = results
        return results

    def _optimize_single_user(self, df_u, uid, n_trials):

        def objective(trial):
            return self._objective(trial, df_u)

        study = optuna.create_study(
            directions=list(self.directions),
            study_name=f"user_{uid}",
            sampler=optuna.samplers.TPESampler(
                                                n_startup_trials=50,
                                                multivariate=True,
                                                group=True
                                            )
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=self.n_jobs
        )

        pareto = [
            {
                "params": t.params,
                "predictability": t.values[0],
                "entropy": t.values[1]
            }
            for t in study.best_trials
        ]
        return pareto, study

    # -----------------------------------------------------------
    # CHOOSE BEST PARAMS FROM PARETO
    # -----------------------------------------------------------
    def select_best(self):
        """
        Select the best Pareto solution for a given user_id using
        normalized scalarization:
            - predictability: maximize
            - entropy: maximize
        Combined score = x_norm + y_norm
        """
        processed_data = []
        for user_id in self.df.index.unique():
            pareto_list = self.pareto[user_id]["pareto"]

            if len(pareto_list) == 0:
                raise ValueError(f"Pareto front for user_id={user_id} is empty.")

            # Convert to DataFrame
            pf = pd.DataFrame(pareto_list)

            # Rename metrics for clarity (x = predictability, y = entropy)
            pf = pf.rename(columns={"predictability": "x", "entropy": "y"})

            # 1. Normalize x and y (avoid zero division)
            x_min, x_max = pf["x"].min(), pf["x"].max()
            y_min, y_max = pf["y"].min(), pf["y"].max()

            # Predictability (x): maximize → normalized upward
            x_norm = (pf["x"] - x_min) / (x_max - x_min + 1e-9)

            # Entropy (y): maximize → normalized upward
            y_norm = (pf["y"] - y_min) / (y_max - y_min + 1e-9)

            # 2. Combined score
            pf["score"] = x_norm + y_norm

            # 3. Select best index
            best_row = pf.loc[pf["score"].idxmax()]

            self.picks[user_id] = {
                "params": best_row["params"],
                "predictability": float(best_row["x"]),
                "entropy": float(best_row["y"]),
                "pick_score": float(best_row["score"]),
            }
            processed_data.append(self._run_detector(self.df.loc[user_id],best_row["params"]))
        self.best_pick_data = pd.concat(processed_data)
        return self.best_pick_data


df = pd.read_csv(r"D:\GitHub\OptimalStop\synthetic_ground_truth\random_city.csv").iloc[:, 1:]
df.columns = ['lat', 'lon', 'datetime', 'user_id', 'label']
df = df[['lat', 'lon', 'datetime', 'user_id']]
df = df[df['user_id'] < 4]
df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lat, df.lon), crs=3857)  # only synthetic
ostop = OptimalStop(df=df, detector='lachesis', param_space={
    'eps': (5, 50),
    'min_samples': (1, 3),
    'stop_distance': (5, 50),
    'stop_time': (300, 3000)
})
ostop.fit_predict(200)
ostop.select_best()
