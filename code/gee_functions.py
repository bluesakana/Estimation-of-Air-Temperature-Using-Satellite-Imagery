## Functions for processing Google Earth Engine Images

import folium
import numpy as np
import pandas as pd
import geopandas as gpd
import ee
import geemap
from settings import *

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import requests

def get_s2_w_cloud_prob(aoi, start_date, end_date):
    '''
    Function that merges 2 collections of SENTINEL 2
    '''
    # get the actual image in all bands
    sen2 = (ee.ImageCollection('COPERNICUS/S2_SR')
            .filterDate(start_date, end_date)
            .filterBounds(aoi)
            .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER))
            #.select(['B4', 'B3', 'B2'])
           )

    # get the cloud probability scale
    sen2_cloud = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                 .filterDate(start_date, end_date)
                 .filterBounds(aoi)
                 )
    
    # define merging parameters
    merge_params = {'primary': sen2,
                        'secondary': sen2_cloud,
                        'condition': ee.Filter.equals(**{'leftField': 'system:index', 'rightField': 'system:index'})
                       }
    
    # merge the 2 images together
    sen2_cloudless = ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**merge_params))
                 
    return sen2_cloudless


def add_cloud_bands(img):
    '''
    Function that adds the cloud probability layer
    '''
    # subset the probability
    cloud_prob = ee.Image(img.get('s2cloudless')).select('probability')
    
    # filter only those that pass the threshold
    is_cloud = cloud_prob.gt(CLOUD_PROB_THRES).rename('clouds')
    
    # add a new variable
    return img.addBands(ee.Image([cloud_prob, is_cloud]))
    
    
def add_shadow_bands(img):
    '''
    Function that adds the shadow probability layer
    '''
    # identify non-water pixels
    not_water = img.select('SCL').neq(6)
    
    # identify dark near-infra that are not water
    sr_band_scale = 1e4
    dark_pixels = img.select('B8').lt(NIR_DARK_THRES*sr_band_scale).multiply(not_water).rename('dark_pixels')
    
    # determine direction to project cloud shadow from clouds
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')))
    
    # project shadows from clouds for distance from clouds
    cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, CLOUD_PROJ_DIST*10)
                .reproject(**{'crs': img.select(0).projection(), 'scale': SEN2_SCALE})
                .select('distance')
                .mask()
                .rename('cloud_transform')
               )
    
    # identify intersection of dark pixels with cloud shadow projection
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')
    
    # add to main img
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))


def add_cloud_shadow_mask(img):
    '''
    Function that assemble the cloud and shadow components into a single mask
    '''
    # add cloud band
    img_cloud = add_cloud_bands(img)
    
    # add shadows
    img_cloud_shadows = add_shadow_bands(img_cloud)
    
    # combine, set cloud and shadow as value 1, else 0
    is_cld_shdw = img_cloud_shadows.select('clouds').add(img_cloud_shadows.select('shadows')).gt(0)
    
    # remove small cloud patches and dilate remaining pixels by buff input
    is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(BUFFER*2/SEN2_SCALE) # 10m scale
                   .reproject(**{'crs': img.select([0]).projection(), 'scale':SEN2_SCALE})
                   .rename('cloudmask')
                  ) 
    
    # add final cloud mask
    return img_cloud_shadows.addBands(is_cld_shdw)
    
    
def apply_cld_shdw_mask(img):
    '''
    Function that apply cloud mask to each image in collection
    '''
    # Subset the cloudmask band and invert it so clouds/shadow are 0, else 1.
    not_cld_shdw = img.select('cloudmask').Not()

    # Subset reflectance bands and update their masks, return the result.
    return img.select('B.*').updateMask(not_cld_shdw)


def add_ee_layer(self, ee_image_object, vis_params, name, show=True, opacity=1, min_zoom=0):
    '''
    Function that displays Earth Engine image tiles on folium map
    '''
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
        name=name,
        show=show,
        opacity=opacity,
        min_zoom=min_zoom,
        overlay=True,
        control=True
        ).add_to(self)
    
# Add the Earth Engine layer method to folium
folium.Map.add_ee_layer = add_ee_layer


def ee_array_to_df(arr, list_of_bands):
    '''
    Function that transforms client-side ee.Image.getRegion array to pandas.DataFrame.
    '''
    
    df = pd.DataFrame(arr)

    # Rearrange the header.
    headers = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns=headers)

    # Remove rows where list_of_bands are null.
    df = df.dropna(axis=0, subset=list_of_bands)
    # Keep only needed variables
    df = df[['longitude', 'latitude', 'time', 'View_Time', *list_of_bands]]
    
    # Convert the data to numeric values.
    for band in list_of_bands:
        df[band] = pd.to_numeric(df[band], errors='coerce')
    
    # Convert the time field into a datetime.
    time = df['View_Time'].map(lambda x: timedelta(hours=float(x)) if x is not None else None)
    df['datetime'] = pd.to_datetime(df['time'], unit='ms') + time
    
    # Keep the columns of interest.
    df = df[['datetime', 'longitude', 'latitude', *list_of_bands]]

    return df


def kelvin_to_degc(df):
    '''
    Function that converts Kelvins to DegC
    '''
    return df-273.15


def reproject_image(img):
    '''
    Function that reprojects an image
    '''
    return img.reproject(crs=CRS, scale=SEN2_SCALE)


def get_lst(imgCol, poi):
    '''
    Function that gets the list of valid LST measurements from ImageCollection for the specific location from Earth Engine
    '''

    # get all data from point of interest
    lst = (imgCol.getRegion(poi, scale=SEN2_SCALE)
           .getInfo()
          )
    #print(pd.DataFrame(lst))
    # convert to dataframe
    lst = ee_array_to_df(lst, ['LST_1KM'])
    
    # convert to degC
    lst['LST'] = kelvin_to_degc(lst['LST_1KM'])

    # reset index just in case
    lst.reset_index(inplace=True)
    
    # drop useless columns
    lst.drop(['LST_1KM', 'index'], axis=1, inplace=True)
    
    return lst


def get_tile(centroid, size):
    '''
    Function that returns the Earth Engine polygon (square tile) of dimensions size*size 
    '''
    
    return centroid.buffer(size/2).bounds()


def get_sen2_composite_img(imgCol, dt):
    '''
    Function that gets the Sentinel-2 composite image
    '''
    # subtract 9 months from dt
    dt_start = datetime.strftime(datetime.strptime(FINAL_DATE, '%Y-%m-%d')-relativedelta(months=9), '%Y-%m-%d')

    # subset satellite 
    satellite = (imgCol.filterDate(dt_start, FINAL_DATE) # subset to time frame of 9 months
                 #.filterBounds(get_tile(poi, SEN2_TILESIZE)) # subset to just the tile of interest
                )

    # construct cloudfree composite
    tile = (satellite.map(add_cloud_shadow_mask)
                     .map(apply_cld_shdw_mask)
                     .mean()
                     .reproject(crs=CRS, scale=SEN2_SCALE)
           )

    # select only the bands of interest
    tile = tile.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])
        
    return tile


def convert_img_to_np(img, aoi):
    '''
    Function that converts the Earth Engine image object to numpy array
    '''
    
    return geemap.ee_to_numpy(img, region=aoi)


def get_airtemp(station_id, dt):
    '''
    Function that retrieves the air temperature for the given station_id and datetime
    '''
    # retrieve data from data.gov.sg
    date_params = {'date_time': datetime.strftime(dt, "%Y-%m-%dT%H:%M:%S")}
    airtemp_dict = requests.get(WEATHER_URL, date_params).json()

    # take only specified station
    
    airtemp = pd.DataFrame(airtemp_dict['items'][0]['readings'])
    try:
        airtemp = airtemp.loc[airtemp['station_id'] == station_id, 'value'].values[0]
    except:
        airtemp = None

    return airtemp


def show_map(img, center, bands, layer_name, pix_min=0, pix_max=4000):
    '''
    Function that displays an interactive folium map centred around Singapore
    '''
    # Create a folium map object.
    m = folium.Map(location=center, zoom_start=12)

    # Add layers to the folium map.
    m.add_ee_layer(img,
                    {'bands': bands, 'min': pix_min, 'max': pix_max, 'gamma': 1.1},
                    layer_name, True, 1, 9)

    # Add a layer control panel to the map.
    m.add_child(folium.LayerControl())

    # Display the map.
    display(m)
    
    # return the map object
    return m
    
    
def get_centroid_coord(poly):
    '''
    Function that returns the coordinates of the centroid of a polygon in string format
    '''
    return poly.centroid(10).coordinates().reverse().getInfo()


def Mapdisplay(center, dicc, Tiles="OpensTreetMap",zoom_start=12):
    '''
    :param center: Center of the map (Latitude and Longitude).
    :param dicc: Earth Engine Geometries or Tiles dictionary
    :param Tiles: Mapbox Bright,Mapbox Control Room,Stamen Terrain,Stamen Toner,stamenwatercolor,cartodbpositron.
    :zoom_start: Initial zoom level for the map.
    :return: A folium.Map object.
    '''
    mapViz = folium.Map(location=center,tiles=Tiles, zoom_start=zoom_start)
    for k,v in dicc.items():
        if ee.image.Image in [type(x) for x in v.values()]:
            folium.TileLayer(
                tiles = v["tile_fetcher"].url_format,
                attr  = 'Google Earth Engine',
                overlay =True,
                name  = k
              ).add_to(mapViz)
        else:
            folium.GeoJson(
            data = v,
            #style_function = lambda x: {'fillColor': '#FF0000' if x['properties']['airtemp'] > 30 else '#0000FF'},
            #style_function = {'fillColor': 'FF0000'},
            name = k
              ).add_to(mapViz)
            
    mapViz.add_child(folium.LayerControl())
    
    return mapViz


def merge_dfs(dictionary):
    '''
    Function that merges a dictionary of dataframes together by columns, using common columns
    '''
    for i, key in enumerate(dictionary.keys()):
        # special treatment for first dataframe
        if i == 0: df = dictionary[key]
        # for other items, merge
        else: df = pd.merge(df, dictionary[key], validate='one_to_one')
        
    return df


def convert_df_to_np(df):
    '''
    Function that takes in a dataframe and converts each row into a np.array stored in a list, returns a pandas Series
    '''
    
    return df.apply(lambda row: [convert_row_to_np(row)], axis=1 # apply lambda function to each row
                   )


def convert_row_to_np(row):
    '''
    Function that converts a pandas Series into a np.array
    '''
    return np.stack(row.map(np.array), axis=2)