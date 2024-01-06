# https://blog.csdn.net/qq_40206371/article/details/134698358
import folium
import pandas as pd


def visualize_sensor(dataset, key_index=None, indices=None, filename=None, ext=None):
    if dataset.lower() == 'metr_la':
        filepath = f'libcity/cache/graph_sensor_locations.csv'
        df = pd.read_csv(filepath)
    else:
        filepath = 'libcity/cache/graph_sensor_locations_bay.csv'
        df = pd.read_csv(filepath, names=['sensor_id', 'latitude', 'longitude'])
        df.insert(0, 'index', range(len(df)), allow_duplicates=False)

    if indices is None:
        indices = range(len(df))

    mean_latitude = df['latitude'].mean()
    mean_longitude = df['longitude'].mean()

    m = folium.Map(location=(mean_latitude, mean_longitude), zoom_start=12)

    for data in df.iterrows():
        tmp_index = int(data[1]['index'])
        if tmp_index not in indices:
            continue
        tmp_latitude = data[1]['latitude']
        tmp_longitude = data[1]['longitude']
        tmp_sensor_id = int(data[1]['sensor_id'])
        icon = folium.Icon(color='red') if tmp_index == key_index else None
        testing_result = '' if ext is None else ext[tmp_index]
        folium.Marker(location=(tmp_latitude, tmp_longitude), tooltip=f'{tmp_index}',
                      popup=f'{tmp_sensor_id}={testing_result}:({tmp_latitude},{tmp_longitude})', icon=icon).add_to(m)

    if filename is None:
        m.save(f'map_{dataset}.html')
    else:
        m.save(filename)


def visualize_sensor_varying(dataset, key_index, filename=None, ext=None, adjust=False):
    if dataset.lower() == 'metr_la':
        filepath = f'libcity/cache/graph_sensor_locations.csv'
        df = pd.read_csv(filepath)
    else:
        filepath = 'libcity/cache/graph_sensor_locations_bay.csv'
        df = pd.read_csv(filepath, names=['sensor_id', 'latitude', 'longitude'])
        df.insert(0, 'index', len(df), allow_duplicates=False)

    mean_latitude = df['latitude'].mean()
    mean_longitude = df['longitude'].mean()

    if adjust:
        sorted_items = sorted(ext.items(), key=lambda x: x[1])
        minVal = 1.1 * sorted_items[1][1] - 0.1 * sorted_items[-1][1]
        assert minVal > 0, f'diff={sorted_items[-1][1]}-{sorted_items[1][1]}'
        ext[sorted_items[0][0]] = minVal
        diff = max(ext.values()) - minVal
        ext = {key: 1 - (value - minVal) / diff for key, value in ext.items()}

    colormap = folium.LinearColormap(colors=['blue', 'red'], vmin=min(ext.values()), vmax=max(ext.values()))
    m = folium.Map(location=(mean_latitude, mean_longitude), zoom_start=12)

    for data in df.iterrows():
        tmp_index = int(data[1]['index'])
        tmp_latitude = data[1]['latitude']
        tmp_longitude = data[1]['longitude']
        tmp_sensor_id = int(data[1]['sensor_id'])
        if tmp_index == key_index:
            # folium.Marker(location=(tmp_latitude, tmp_longitude), tooltip=f'{tmp_index}',
            #               popup=f'{tmp_sensor_id}:({tmp_latitude},{tmp_longitude})',
            #               icon=folium.Icon(color='red')).add_to(m)
            folium.CircleMarker(location=[tmp_latitude, tmp_longitude],
                                radius=10,
                                color='blue',
                                fill=True,
                                fill_color='blue',
                                fill_opacity=1,
                                popup=f'{tmp_sensor_id}:({tmp_latitude},{tmp_longitude})').add_to(m)
        else:
            folium.CircleMarker(location=[tmp_latitude, tmp_longitude],
                                radius=10,
                                color=colormap(ext[tmp_index]),
                                fill=True,
                                fill_color=colormap(ext[tmp_index]),
                                fill_opacity=1,
                                popup=f'{tmp_sensor_id}:({tmp_latitude},{tmp_longitude})').add_to(m)

    # Add LinearColormap to the map
    colormap.caption = 'Bayesian Evidence'
    m.add_child(colormap)

    # Add LayerControl to show/hide the color bar
    folium.LayerControl().add_to(m)

    if filename is None:
        m.save(f'map_{dataset}.html')
    else:
        m.save(filename)


if __name__ == '__main__':
    visualize_sensor('metr_la')
    visualize_sensor('pems_bay')
