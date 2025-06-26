import geopandas as gpd
import os
import pandas as pd

for r,d,f in os.walk(r'D:\GitHub\OptimalStop'):
    for ff in f:
        if '.shp' in ff and 'info' in ff:
            if '7' in ff or '8' in ff:
                shppath = os.path.join(r,ff)
                shapefile = gpd.read_file(shppath)
                shapefile = shapefile[['field_5', 'field_6', 'geometry']]
                shapefile['x'] = shapefile.geometry.apply(lambda geom: geom.x)
                shapefile['y'] = shapefile.geometry.apply(lambda geom: geom.y)
                shapefile['field_5'] = pd.to_datetime(shapefile['field_5'])
                shapefile.to_csv(ff.split('.')[0]+'_repair.csv')
