import geopandas as gpd
import os
import pandas as pd
import numpy as np
from geopy.distance import geodesic

df = pd.read_csv('data/NZ_example.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('user_id')
df = df[['datetime', 'lat', 'lon']]
df.columns = ['datetime', 'lon', 'lat']
df.index = df.index.map({v: k for k, v in enumerate(pd.unique(df.index))})
df = df.loc[:20]
df = df.reset_index()

for r,d,f in os.walk(r'D:\GitHub\OptimalStop\data'):
    for ff in f:
        if '.shp' in ff and 'info' in ff:
            shppath = os.path.join(r,ff)
            shapefile = gpd.read_file(shppath)
            shapefile = shapefile[['field_5','field_6','geometry']]
            shapefile['x'] = shapefile.geometry.apply(lambda geom: geom.x)
            shapefile['y'] = shapefile.geometry.apply(lambda geom: geom.y)
            shapefile['field_5'] = pd.to_datetime(shapefile['field_5'])
            shapefile.reset_index(drop=True,inplace=True)
            uid = int(ff.split('_')[1].split('.')[0])
            current_user = df[df['user_id'] == uid]
            current_user.reset_index(drop=True, inplace=True)
            current_user['datetime'] = pd.to_datetime(current_user['datetime'])
            new_df = []
            for u, group in shapefile.groupby(['field_5','field_6']):
                if group.shape[0] > 1:
                    this_time = group.field_5.values[0]
                    corresponding = current_user[current_user.datetime == group.field_5.values[0]].reset_index(drop=True)
                    group = group.reset_index(drop=True)
                    if corresponding.shape[0] != group.shape[0] or (corresponding.lon != group.x).any() or (corresponding.lat != group.y).any():
                        larger_corresponding = current_user[current_user.lon.isin(group.x) & current_user.lat.isin(group.y)].reset_index(drop=True)
                        if larger_corresponding.shape[0] != group.shape[0] and group.field_6.values[0] != -1:
                            datetime_location = (larger_corresponding.datetime == group.field_5.values[0]).idxmax()
                            merged = pd.concat([group.shift(datetime_location),larger_corresponding],axis=1)
                            merged = merged[~merged.isna().any(axis=1)]
                            if (merged.x != merged.lon).any() or (merged.y != merged.lat).any():
                                merged
                            else:
                                merged['field_5'] = merged['datetime']
                                merged = merged[['field_5','field_6','geometry','x','y']]
                                new_df.append(merged)
                        else:
                            joined = pd.merge(group,larger_corresponding,left_on=['x','y'],right_on=['lon','lat']).sort_values('datetime')
                            joined['field_5'] = joined['datetime']
                            joined = joined[['field_5','field_6','geometry','x','y']]
                            new_df.append(joined)
                    else:
                        group['field_5'] = corresponding['datetime']
                        new_df.append(group)
                else:
                    new_df.append(group)
            new_df = pd.concat(new_df).drop_duplicates().sort_values('field_5')
            new_df.reset_index(drop=True,inplace=True)
            new_df.to_csv(f'infostop_{uid}_repair.csv')
