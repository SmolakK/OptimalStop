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
            conca = pd.concat((shapefile, current_user.iloc[1:].reset_index(drop=True)), axis=1)
            conca['inequal'] = conca.x == conca.lon
            conca['swaps'] = (conca['inequal'] != conca['inequal'].shift()).cumsum()
            new_df = []
            for swap_id,swapped in conca.groupby('swaps'):
                if swapped['inequal'].all() != True:
                    shapefile_now = swapped[shapefile.columns]
                    current_now = swapped[current_user.columns]
                    reshaped = shapefile_now.merge(current_now,left_on=['x','y'],right_on=['lon','lat'],how='outer')
                    new_df.append(reshaped[list(shapefile.columns)+['datetime']])
                else:
                    new_df.append(swapped[list(shapefile.columns)+['datetime']])
            # check
            new_df = pd.concat(new_df)
            new_df = new_df = new_df.rename({'datetime':'datetime_2'},axis=1)
            conca_again = pd.concat((new_df.reset_index(drop=True), current_user.iloc[1:].reset_index(drop=True)), axis=1)
            new_df['field_5'] = new_df['datetime_2']
            new_df = new_df[['field_5','field_6','geometry','x','y']]
            new_df.to_csv(f'infostop_{uid}_repair.csv')
