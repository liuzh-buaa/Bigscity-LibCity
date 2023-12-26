# https://blog.csdn.net/qq_40206371/article/details/134698358
import pandas as pd
import folium

filepath = 'libcity/cache/graph_sensor_locations.csv'
# filepath = 'libcity/cache/graph_sensor_locations_bay.csv'

df = pd.read_csv(filepath, names=['station_id', 'lat', 'lon'])
