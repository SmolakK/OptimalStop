import geopandas as gpd
import os
import pandas as pd
import geopandas as gpd

to_concat = {}
for r,d,f in os.walk(r'D:\GitHub\OptimalStop\data'):
    for ff in f:
        if '.csv' in ff and 'repair' in ff:
            print(ff)
            uid = int(ff.split('_')[1])
            to_concat[uid] = pd.read_csv(os.path.join(r,ff))
        if '.shp' in ff and int(ff.split('_')[1].split('.')[0]) >= 9 and int(ff.split('_')[1].split('.')[0]) <= 19:
            uid = int(ff.split('_')[1].split('.')[0])
            curshp = gpd.read_file(os.path.join(r,ff))
            curshp = curshp.loc[:,curshp.columns[1:]]
            curshp.rename({'field_3':'y','field_4':'x','field_2':'Unnamed: 0'},axis=1,inplace=True)
            to_concat[uid] = curshp
reference = pd.concat(to_concat).droplevel(1).reset_index()
reference = reference[['index','field_5','field_6','geometry','x','y']]
reference.columns = ['user_id','datetime','labels','geometry','lon','lat']
reference = reference.drop_duplicates()
reference['datetime'] = pd.to_datetime(reference['datetime'])
missing_mask = reference['datetime'].isna()
reference['datetime'] = reference['datetime'].fillna(method='ffill')
reference['datetime'] += pd.to_timedelta(reference['datetime'].duplicated().cumsum(), unit='s')
reference['datetime'][missing_mask] = reference['datetime'][missing_mask] + pd.to_timedelta('1s')
len(reference.groupby('user_id'))  # users
reference.groupby('user_id').apply(lambda x: len(pd.unique(x.labels[x.labels != -1]))).sum()  # stops
reference_stops = reference[reference.labels != -1]
reference_stops['moved'] = reference_stops.groupby('user_id').apply(lambda x: (x.labels != x.labels.shift(-1)).cumsum()).droplevel(0)
reference_stops.groupby('user_id').moved.max().sum()  # stop events
reference.to_csv(r'data\reference.csv')