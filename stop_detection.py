import pandas as pd
import numpy as np
from infostop import Infostop
import concurrent.futures as cf
from itertools import repeat
from shapely import Point


class ClusteringAggregator():
    """
	A class for spatial aggregation of movement trajectories. This one uses clustering algorithms from sklearn library.
	"""

    def __init__(self, algorithm, stop_distance, stop_time, **kwargs):
        """
		Class initialisation. Accepts sklearn clustering algorithms' classes and their keyword arguments.

		Args:
			algorithm: Clustering algorithm from sklearn library.
			**kwargs: Any accepted kwargs.
		"""
        self._algorithm = algorithm(**kwargs)
        self.stop_distance = stop_distance
        self.stop_time = stop_time

    @property
    def algorithm(self):
        return self._algorithm

    def _user_aggregate(self, single_trajectory):
        """
		Spatially aggregates single movement trajectory. Adds labels columns.

		Args:
			single_trajectory: Single movement trajectory

		Returns:
			An aggregated TrajectoriesFrame with labels column added.
		"""
        try:
            fited = self.algorithm.fit(single_trajectory)
        except:
            raise ValueError
        single_trajectory['labels'] = fited.labels_
        return single_trajectory

    def _recalcuate_centres(self, single_trajectory):
        """
		Based on labels, recalculates spatial coordinates of points to their clusters centers.

		Args:
			single_trajectory: single movement trajectory

		Returns:
			a TrajectoriesFrame with overwritten coordinates in lon and lat columns
		"""
        centres = single_trajectory.groupby(by='labels').apply(lambda x: x.mean())
        single_trajectory = single_trajectory.join(centres, on='labels', rsuffix='_')
        single_trajectory['lat'] = single_trajectory['lat_']
        single_trajectory['lon'] = single_trajectory['lon_']
        return single_trajectory

    def aggregate(self, trajectories_frame, drop_noise=False, centres_as_geometry=False):
        """
		General function for spatial data aggregation. It assigns labels to clusters and has additional options of
		centers recalculation and data overwriting.

		Args:
			trajectories_frame: TrajectoriesFrame object class
			drop_noise: Datermines if noise (-1 labels) should be drop
			centres_as_geometry: If true, centres of clusters are assigned as geometry
		Raturns:
			A spatially aggregated TrajectoriesFrame
		"""
        trajectories_frame_copy = trajectories_frame.copy()
        trajectories_frame_copy = stop_detection(trajectories_frame_copy, distance_condition=self.stop_distance,
                                                 time_condition=self.stop_time)
        stopmask = trajectories_frame_copy.is_stop != -1
        traj_frame = trajectories_frame_copy[stopmask]
        coordinates_frame = traj_frame[['user_id', 'lon', 'lat']]
        clustered = coordinates_frame.groupby('user_id').apply(lambda x: self._user_aggregate(x[['lon', 'lat']]),include_groups=False)
        if centres_as_geometry:
            clustered = clustered.groupby('user_id').apply(lambda x: self._recalcuate_centres(x))
        merged = pd.merge(trajectories_frame_copy, clustered, left_index=True, right_index=True, how='outer')
        merged = merged[[x for x in merged.columns if '_y' not in x and 'level' not in x and 'stop' not in x]]
        merged['labels'] = merged['labels'].fillna(-1)
        merged = merged.rename({'lon_x': 'lon', 'lat_x': 'lat'}, axis=1)
        merged = merged.set_index('user_id')
        return merged


def _user_stops(indi, single_trajectory, distance_condition, time_condition):
    """
	Detects stops in a single user's trajectory

	Args:
		indi: user identifier
		single_trajectory: user's trajectory
		distance_condition: distance threshold for stop detection
		time_condition: time threshold for stop detection

	Returns:
		TrajectoriesFrame with records indicated as stops in 'is_stop' column
	"""
    single_trajectory = single_trajectory.copy()
    single_trajectory['datetime'] = pd.to_datetime(single_trajectory['datetime'])
    single_trajectory.reset_index(inplace=True)
    starting_index = 0
    stops = []
    j = 0
    while starting_index + j < single_trajectory.shape[0] - 1:
        j += 1
        ending_index = starting_index + j
        start_point = Point(single_trajectory.at[starting_index, 'lat'], single_trajectory.at[starting_index, 'lon'])
        end_point = Point(single_trajectory.at[ending_index, 'lat'], single_trajectory.at[ending_index, 'lon'])

        actual_distance = start_point.distance(end_point)
        if actual_distance > distance_condition:
            if ending_index - starting_index > 1:
                start_time = single_trajectory.at[starting_index, 'datetime']
                end_time = single_trajectory.at[ending_index - 1, 'datetime']
                if time_condition:
                    elapsed = end_time - start_time
                    if elapsed > pd.Timedelta(time_condition):
                        stops.append(list(range(starting_index, ending_index)))
                    else:
                        starting_index = ending_index
                        j = 0
                        continue
                else:
                    stops.append(list(range(starting_index, ending_index)))
            starting_index = ending_index
            j = 0
    single_trajectory['is_stop'] = -1
    stop_id = 0

    # After the while loop
    if starting_index < single_trajectory.shape[0] - 1:
        # We have a leftover cluster from starting_index to the last row
        cluster_length = (single_trajectory.shape[0] - starting_index)
        if cluster_length > 1:
            if time_condition:
                start_time = single_trajectory.at[starting_index, 'datetime']
                end_time = single_trajectory.at[single_trajectory.index[-1], 'datetime']
                elapsed = end_time - start_time
                if elapsed > pd.Timedelta(time_condition):
                    stops.append(list(range(starting_index, single_trajectory.shape[0])))
            else:
                stops.append(list(range(starting_index, single_trajectory.shape[0])))

    for stop in stops:
        selected = single_trajectory.iloc[stop]
        # mean_lat = selected['lat'].mean()
        # mean_lon = selected['lon'].mean()
        # single_trajectory.iloc[stop, single_trajectory.columns.get_loc('lat')] = mean_lat
        # single_trajectory.iloc[stop, single_trajectory.columns.get_loc('lon')] = mean_lon
        single_trajectory.iloc[stop, single_trajectory.columns.get_loc('is_stop')] = stop_id
        stop_id += 1
    return indi, single_trajectory.set_index('index')


def stop_detection(trajectories_frame, distance_condition=300, time_condition='10 min'):
    """
	Detects all stops in the TrajectoriesFrame. Uses multithreading.

	Args:
		trajectories_frame: TrajectoriesFrame class object
		distance_condition: distance threshold for stop detection
		time_condition: time threshold for stop detection

	Returns:
		TrajectoriesFrame with records indicated as stops in 'is_stop' column
	"""
    result_dic = {}
    with cf.ThreadPoolExecutor() as executor:
        args = [val for indi, val in trajectories_frame.groupby('user_id')]
        ids = [indi for indi, val in trajectories_frame.groupby('user_id')]
        results = list(executor.map(_user_stops, ids, args, repeat(distance_condition), repeat(time_condition)))
    for result in results:
        result_dic[result[0]] = result[1]
    detected = pd.concat([x for x in result_dic.values()])
    return detected


def convert_to_unix(group):
    group['datetime'] = group['datetime'].apply(lambda x: int(pd.to_datetime(x).timestamp()))
    return group


def convert_from_unix(group):
    group['datetime'] = pd.to_datetime(group['datetime'], unit='s')
    return group


def infostop_single(group, r1=30, r2=30, min_staying_time=600, max_time_between=86400, min_size=2):
    model = Infostop(r1=r1,
                     r2=r2,
                     label_singleton=False,
                     min_staying_time=min_staying_time,
                     max_time_between=max_time_between,
                     min_size=min_size)
    group = group.sort_values('datetime')
    group = group[['lat', 'lon', 'datetime']]
    group = np.array(group)
    labels = model.fit_predict(group)
    group = np.hstack([group, labels.reshape(-1, 1)])
    return pd.DataFrame(group, columns=['lat', 'lon', 'datetime', 'labels'])


def infostop(dataset, r1=30, r2=30, min_staying_time=600, max_time_between=86400, min_size=2):
    dataset = dataset.copy()
    dataset = convert_to_unix(dataset)
    to_concat = {}
    for user_id, group in dataset.groupby('user_id'):
        infostop_group = infostop_single(group, r1, r2, min_staying_time, max_time_between, min_size)
        to_concat[user_id] = infostop_group
    dataset = pd.concat(to_concat)
    dataset.index = dataset.index.droplevel(1)
    dataset = convert_from_unix(dataset)
    return dataset
