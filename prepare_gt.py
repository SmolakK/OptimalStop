import os
from stop_detection import infostop, stop_detection, ClusteringAggregator
import pandas as pd
from sklearn.cluster import DBSCAN
from pyproj import Transformer
import geopandas as gpd
from tqdm import tqdm
tqdm.pandas()

f_path = r'D:\GitHub\OptimalStop\data'
files = []
for r,d,f in os.walk(f_path):
    for ff in f:
        try:
            if '.csv' in ff and int(ff.split('_')[1].split('.')[0]) > 9:
                fullpath = os.path.join(r,ff)
                df = pd.read_csv(fullpath)
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.drop_duplicates()
                df.rename({'index':'user_id'},axis=1,inplace=True)
                df = df.set_index('user_id')
                df.rename({'Unnamed: 0':'index'},axis=1,inplace=True)
                files.append(df)
        except:
            pass
df = pd.concat(files)
transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
df[['lon', 'lat']] = df.apply(
    lambda row: transformer.transform(row['lon'], row['lat']), axis=1,
    result_type='expand')
df = df.reset_index()
clust_agg = ClusteringAggregator(DBSCAN, stop_distance=30, stop_time='5T',
                                 **{"eps": 200, "min_samples": 3})
aggregated_df = clust_agg.aggregate(df)
for uid, udata in aggregated_df.groupby(level=0):
    udata = gpd.GeoDataFrame(udata, geometry=gpd.points_from_xy(udata['lon'], udata['lat']))
    udata['datetime'] = udata.datetime.astype(str)
    udata.rename({'labels_x':'labels'},axis=1,inplace=True)
    udata.to_file(f'data\infostop_{uid}.shp', driver='ESRI Shapefile')