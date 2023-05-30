CLOUD_FILTER = 80
CLOUD_PROB_THRES = 60
NIR_DARK_THRES = 0.15
CLOUD_PROJ_DIST = 1
BUFFER = 50

SEN2_SCALE = 10
SEN2_TILESIZE = 320 # in metres
SEN2_SCALEFACTOR = 0.0001 # normalisation factor for Sentinel 2 raw data
SEN2_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

ORIGIN_DATE = '2017-04-01'
FINAL_DATE = '2023-05-20' # was 13
CRS = 'EPSG:4326'

WEATHER_URL = 'https://api.data.gov.sg/v1/environment/air-temperature'

RANDOM_STATE = 42