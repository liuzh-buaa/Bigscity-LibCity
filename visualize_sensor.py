# https://blog.csdn.net/qq_40206371/article/details/134698358
import pandas as pd
import folium

filepath = 'libcity/cache/graph_sensor_locations.csv'
df = pd.read_csv(filepath)

# filepath = 'libcity/cache/graph_sensor_locations_bay.csv'
# df = pd.read_csv(filepath, names=['sensor_id', 'latitude', 'longitude'])

mean_latitude = df['latitude'].mean()
mean_longitude = df['longitude'].mean()

m = folium.Map(location=(mean_latitude, mean_longitude), zoom_start=12)

for data in df.iterrows():
    tmp_latitude = data[1]['latitude']
    tmp_longitude = data[1]['longitude']
    tmp_sensor_id = data[1]['sensor_id']
    folium.Marker(location=(tmp_latitude, tmp_longitude),
                  popup='{}:({},{})'.format(int(tmp_sensor_id), tmp_latitude, tmp_longitude)).add_to(m)

m.save('map_metr_la.html')
# m.save('map_pems_bay.html')
