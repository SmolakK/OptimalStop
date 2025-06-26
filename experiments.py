import os
import pandas as pd
from stop_detection import infostop, stop_detection, ClusteringAggregator
from tqdm import tqdm
from predictability_metrics import *
from complexity_metrics import *
from sklearn.cluster import DBSCAN
tqdm.pandas()
# from humobi.measures.individual import real_predictability
from pyproj import Transformer
import pandas as pd
from evaluation_metrics import *


def distances(col_name):
    res = []
    for uid,da in reference.groupby(level=0):
        res.append(pd.DataFrame(np.sqrt((x.loc[uid].Pred - reference.loc[uid].Pred) ** 2 + (x.loc[uid][col_name] - reference.loc[uid][col_name]) ** 2) for x in results.values()))
    return res


def mode_label(series):
    return series.mode()[0] if not series.empty else None


df = pd.read_csv('data/reference.csv').iloc[:,1:]
df['datetime'] = pd.to_datetime(df['datetime'])
df.columns = ['user_id','datetime','labels_reference','geometry','lat','lon']
df = df.set_index('user_id')
df = df[['datetime', 'lon', 'lat','labels_reference']]
infostop_clus = clusters_time_entropy(df, 'labels_reference')
df_cal = df[df.labels_reference != -1]
uq_points = df_cal.groupby(level=0).nunique().labels_reference
records = df_cal.groupby(level=0).count().labels_reference
stops = df_cal.groupby(level=0).apply(lambda x: (x.labels_reference != x.labels_reference.shift()).sum())
infostop_re = random_predictability(df_cal, 'labels_reference')[0]

infostop_trans = transition_entropy(df_cal, 'labels_reference')
df_resampled = df_cal.groupby(level=0).apply(lambda x: x[x.labels_reference != x.labels_reference.shift()]).droplevel(1)
df_resampled = df_resampled.rename({'labels_reference':'labels'},axis=1)
infostop_pred, infostop_real = real_predictability(df_resampled)
infostop_shuff = shuffled_entropy(df_resampled)
reference = pd.concat(
    [infostop_pred, infostop_real, infostop_clus, infostop_trans, infostop_re, infostop_shuff, uq_points, records,stops], axis=1)
reference.columns = ['Pred', 'Real', 'Clus', 'Trans', 'Random', 'Shuffled', 'Uq', 'Records', 'Stops']
df = df.reset_index()
transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
df[['lon', 'lat']] = df.apply(lambda row: transformer.transform(row['lon'], row['lat']), axis=1,
                                                result_type='expand')
results = {}
for r1_level in tqdm([50,100,200,400,500,1000,2000,5000],total=5):
    totnumpoint = df.groupby('user_id').apply(lambda x: x.shape[0])
    detected_df = stop_detection(df)
    # detected_df = detected_df[detected_df.is_stop != -1]

    eps = r1_level  # DEFINE SPATIAL UNIT
    min_pts = 3  # OTHER HYPERPARAMETERS
    clust_agg = ClusteringAggregator(DBSCAN,
                                     **{"eps": eps, "min_samples": min_pts})  # DEFINE SPATIAL AGGREGATION ALGORITHM
    df_sel_dbscan = clust_agg.aggregate(detected_df)
    infostop_df = df_sel_dbscan
    # infostop_df = infostop(df,r1=r1_level,r2=r1_level)
    # infostop_df = pd.concat([infostop_df,df.set_index('user_id').labels_reference],axis=1)
    clustering_score = clustering_evaluation(infostop_df)
    VI = variation_of_information(infostop_df)
    over = oversegmentation(infostop_df)
    under = undersegmentation(infostop_df)
    miss = missed(infostop_df)
    eval = clustering_score
    infostop_clus = clusters_time_entropy(infostop_df)
    #
    infostop_df = infostop_df[infostop_df.labels != -1]
    uq_points = infostop_df.groupby(level=0).nunique().labels
    records = infostop_df.groupby(level=0).count().labels
    stops = infostop_df.groupby(level=0).apply(lambda x: (x.labels_reference != x.labels_reference.shift()).sum())
    infostop_re = random_predictability(infostop_df, 'labels')[0]

    infostop_trans = transition_entropy(infostop_df)
    # infostop_df_resampled = pd.DataFrame(infostop_df.groupby(level=0).apply(lambda x: x.set_index('datetime').labels.resample('1H').apply(mode_label)))
    infostop_df_resampled = infostop_df.groupby(level=0).apply(lambda x: x[x.labels != x.labels.shift()]).droplevel(1)
    # infostop_df_resampled = infostop_df
    infostop_trans2 = transitions(infostop_df_resampled)
    infostop_trans3 = transitions(infostop_df_resampled, length=3)
    infostop_trans5 = transitions(infostop_df_resampled, length=5)
    infostop_pred, infostop_real = real_predictability(infostop_df_resampled)
    infostop_shuff = shuffled_entropy(infostop_df_resampled)
    summed = pd.concat([infostop_pred,infostop_real,infostop_clus,infostop_trans,infostop_re, infostop_shuff, uq_points, records, stops, miss, infostop_trans2, infostop_trans3, infostop_trans5, over, under, over/under, eval],axis=1)
    summed.columns = ['Pred','Real','Clus','Trans','Random','Shuffled', 'Uq', 'Records', "Stops", 'Miss','Trans2', 'Trans3','Trans5', 'over', 'under', 'ouratio', 'eval']
    results[r1_level] = summed
[x.idxmin() for x in distances('Clus')]