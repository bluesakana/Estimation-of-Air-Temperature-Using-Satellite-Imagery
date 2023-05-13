## Functions for processing Google Earth Engine Images

import folium
import ee
import geemap
from settings import *

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
                .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
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
    is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(BUFFER*2/10) # 10m scale
                   .reproject(**{'crs': img.select([0]).projection(), 'scale':10})
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
