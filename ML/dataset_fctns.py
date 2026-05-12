import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
from PIL import Image

def columns_to_datetime(ds, columns):
    for colname in columns:
        ds[colname] = pd.to_datetime(ds[colname], format='mixed')
    return ds

def read_phen_dataset(adress, drop_list = []):
    phen_data = pd.read_csv(adress, encoding = "latin1", engine='python', sep = r';\s+|;\t+|;\s+\t+|;\t+\s+|;|\s+;|\t+;|\s+\t+;|\t+\s+;')
    for drop_name in drop_list:
        phen_data = phen_data.drop(drop_name, axis = 1)
    
    ## CONVERT DATE TO DATETIME ##
    phen_data['Eintrittsdatum'] = pd.to_datetime(phen_data['Eintrittsdatum'], format = '%Y%m%d')
    return phen_data

def order_phen_dataset(phen_data):
    ## SORT BY TIME ##
    return phen_data.sort_values(by = ['Stations_id', 'Referenzjahr', 'Eintrittsdatum', 'Order of phase'])

def time_to_next_stage(phen_data):
    #Note phen_data must be time and station ordered. Only plots time to next stage - naive as doesn't consider missing phases.
    ## CALCULATE TIME TO NEXT STAGE ##
    phen_data['Time to next stage'] = phen_data['Eintrittsdatum'].shift(-1) - phen_data['Eintrittsdatum']
    phen_data['Next stage name'] = phen_data['Name of phase'].shift(-1)
    ## EXCLUDE CHANGES IN STATION ##
    phen_data.loc[phen_data['Stations_id'] != phen_data['Stations_id'].shift(-1), 'Time to next stage'] = np.nan
    phen_data.loc[phen_data['Stations_id'] != phen_data['Stations_id'].shift(-1), 'Next stage name'] = np.nan
    return phen_data

def get_phase_name(phaseid, ds_phase_names):
    return ds_phase_names['Phase_englisch'][ds_phase_names['Phase_ID'] == str(phaseid)].values[0]

def get_station_locations(dataset, ds_stations, check_lists = True):
    ds_stations.index = ds_stations['Stations_id']
    lat = [ds_stations._get_value(row, col) for row, col in zip(dataset['Stations_id'], ['geograph.Breite' for count in range(len(dataset))])] #station_data.lookup(row_labels = dataset['Stations_id'], col_labels = ['geograph.Breite'])
    lon = [ds_stations._get_value(row, col) for row, col in zip(dataset['Stations_id'], ['geograph.Laenge' for count in range(len(dataset))])] #station_data._lookup(dataset['Stations_id'], ['geograph.Laenge'])
    dataset['lat'] = lat
    dataset['lon'] = lon
    if check_lists:
        dataset['lat'] = dataset['lat'].map(lambda x: x[0] if isinstance(x, np.float64) == False else x)
        dataset['lon'] = dataset['lon'].map(lambda x: x[0] if isinstance(x, np.float64) == False else x)
    return dataset

def add_locations(dataset, ds_stations):
    LAT, LON = get_station_locations(dataset, ds_stations)
    dataset['lat'] = LAT
    dataset['lon'] = LON
    dataset['lat'] = dataset['lat'].map(lambda x: x[0] if isinstance(x, np.float64) == False else x)
    dataset['lon'] = dataset['lon'].map(lambda x: x[0] if isinstance(x, np.float64) == False else x)
    return dataset

def phase_order_name(dataset, ds_phase_names, stage_order): #[10, 12, 67, 65, 5, 6, 19, 20, 21, 24, ]
    dataset['Order of phase'] = np.nan
    dataset['Name of phase'] = ''
    for i, phaseid in enumerate(stage_order):
        if len(ds_phase_names['Phase_englisch'][ds_phase_names['Phase_ID'] == str(phaseid)]) != 0:
            #print(i, phaseid)
            dataset.loc[dataset['Phase_id'] == phaseid, 'Order of phase'] = i
            dataset.loc[dataset['Phase_id'] == phaseid, 'Name of phase'] = get_phase_name(phaseid, ds_phase_names)
    return dataset

def isolate_stage(phen_data, stage):
    return phen_data[phen_data['Name of phase'] == stage]

def time_stage_to_stage(phen_data, stage1, stage2, winter_sowing = False):
    stage1_frame = isolate_stage(phen_data, stage1)
    if winter_sowing: #If the first stage is actually in winter of the previous year, compare the first stage to the year after.
        stage1_frame.loc[:, 'Referenzjahr'] = stage1_frame.loc[:, 'Referenzjahr'] + 1
    stage1_frame.set_index(['Referenzjahr', 'Stations_id'], inplace=True)
    stage2_frame = isolate_stage(phen_data, stage2)
    stage2_frame.set_index(['Referenzjahr', 'Stations_id'], inplace=True)
    print()
    return (stage2_frame['Eintrittsdatum'] - stage1_frame['Eintrittsdatum'])/ pd.to_timedelta(1, unit='D') #.astype(np.float64)

def interpolate_xy(x, y, ds):
    #Interpolates the input array onto the (non-gridded e.g. phenology station) coordinates x and y.
    #Note hyras is not stored on the full grid, but on some kind of subset. Not quite sure how this works. Just got to hope the stations are in a hyras gridpoint.
    X_for_interp = xr.DataArray(x, dims="modelpoint")
    Y_for_interp = xr.DataArray(y, dims="modelpoint")
    return ds.interp(x=X_for_interp, y=Y_for_interp)#, kwargs={"fill_value": None})

def latlon_to_projection(x_coords, y_coords, epsg_num = 3034):
    proj_epsg = ccrs.epsg(epsg_num)
    proj_latlon = ccrs.PlateCarree()
    points_epsg = proj_epsg.transform_points(proj_latlon, x_coords, y_coords)
    x_epsg = points_epsg[:, 0]
    y_epsg = points_epsg[:, 1]
    return x_epsg, y_epsg

def WC_SOS(lon, lat):
    #y = np.int32((2*lat) + 286/2)
    #x = np.int32((2*lon) + 720/2)
    y = np.int32((1 - (lat + 59)/143)*286)
    x = np.int32((lon + 180)*2)
    photo = Image.open("Useful_Files\\M1_SOS_WGS84.tif")
    data = np.array(photo)
    return data[y, x]
def WC_EOS(lon, lat):
    #y = np.int32((2*lat) + 286/2)
    #x = np.int32((2*lon) + 720/2)
    y = np.int32((1 - (lat + 59)/143)*286)
    x = np.int32((lon + 180)*2)
    photo = Image.open("Useful_Files\\M1_EOS_WGS84.tif")
    data = np.array(photo)
    return data[y, x]

def add_SOS_to_df(df):
    df.loc[:, 'SOS'] = WC_SOS(df['lon'], df['lat'])
    return df
def add_EOS_to_df(df):
    df.loc[:, 'EOS'] = WC_EOS(df['lon'], df['lat'])
    return df

