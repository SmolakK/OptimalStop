import os
import pandas as pd
import sys
sys.path.append(r'D:\GitHub\HuMobi\src\humobi')
from structures.trajectory import TrajectoriesFrame
from tqdm import tqdm
import geopandas as gpd


def count_days(group):
    hours = 30*24
    rsmpl = group.resample('1H', on='datetime').lat.count()
    rsmpl[rsmpl > 0] = 1
    roll = rsmpl.rolling(hours).apply(lambda x: x.sum() / hours)
    if roll.max() >= .85:
        return roll.idxmax()


def filter_dataset(data):
    cnt = 0
    hours = 30*24
    data['datetime'] = data.index.get_level_values(1)
    for uid, group in tqdm(data.groupby(level=0)):
        rsmpl = group.resample('1H', on='datetime').lat.count()
        rsmpl[rsmpl > 0] = 1
        roll = rsmpl.rolling(hours).apply(lambda x: x.sum() / hours)
        to_concat = []
        moves = (group.shift().distance(group) >= 500).sum()
        if roll.max() >= .85 and moves >= 1:
            cnt += 1
            print(cnt)
            end = roll.idxmax()
            start = end - pd.Timedelta(f'{hours} hours')
            group = group[(group['datetime'] >= start) & (group['datetime'] <= end)]
            group = group.sort_index()
            to_concat.append(group)
    data = pd.concat(to_concat)
    return data

# NZ DATA NZ_active - At least 60% of data in hourly intervals for 30 days
fpath = r'Z:\NZ_DATA\NZ_users'
filename = 'NZ_active.csv'
fpath_full = os.path.join(fpath,filename)

TOREAD = 1_000_000

df = pd.read_csv(fpath_full,names=['user_id','lon','lat','datetime'],skiprows=1)
df = gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(*[df[x] for x in ['lat','lon']]),crs=4326)
df.crs = "epsg:4326"
df = df.to_crs(3857)
df.lon = df.geometry.x
df.lat = df.geometry.y
df = TrajectoriesFrame(df,{'crs':3857})
df_filtered = filter_dataset(df)
df_filtered = df_filtered.drop(['datetime'],axis=1).reset_index()
df_filtered.to_parquet('NZ_filtered.csv')

# RIO DATA Second_filter -> 60% over 21 days per 1H
fpath = r'Z:\RJ_DATA\RIO\RIO_active'
filename = r'RIO_second_filter.csv'
fpath_full = os.path.join(fpath,filename)
df = pd.read_csv(fpath_full,names=['user_id','lon','lat','datetime'],skiprows=1)
df = gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(*[df[x] for x in ['lat','lon']]),crs=4326)
df.crs = "epsg:4326"
df = df.to_crs(3857)
df.lon = df.geometry.x
df.lat = df.geometry.y
df = TrajectoriesFrame(df,{'crs':3857})
df_filtered = filter_dataset(df)
df_filtered = df_filtered.drop(['datetime'],axis=1).reset_index()
df_filtered.to_parquet('RIO_filtered.csv')
