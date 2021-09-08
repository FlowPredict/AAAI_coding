import pandas as pd
import numpy as np
import time
from IPython import embed


def cal_location(routes, cur_time):
    if type(routes) == str:
        routes = eval(routes)
    select_step = -1
    for i in range(len(routes)-1):
        if routes[i][0] <= cur_time and routes[i+1][0] > cur_time:
            select_step = i
            break
    if select_step == -1 and cur_time == routes[-1][0]:
        return int(routes[-1][1]), int(routes[-1][2])
    if select_step == -1:
        return -1, -1
    scale = (cur_time - routes[select_step][0])/(routes[select_step+1][0] - routes[select_step][0])
    cur_long = float(routes[select_step][1]) + scale * (float(routes[select_step+1][1]) - float(routes[select_step][1]))
    cur_lat = float(routes[select_step][2]) + scale * (float(routes[select_step+1][2]) - float(routes[select_step][2]))
    return cur_long, cur_lat

def calculate_locations_all(df_real, cur_time):

    df_select = df_real[(df_real['min_route_time'] <= cur_time) & (df_real['max_route_time'] >= cur_time)]
    # df_select = df_real[(df_real['departure_time'] <= cur_time) & (df_real['arrival_time'] >= cur_time)]
    all_flights_locations = []
    for index, row in df_select.iterrows():
        long, lat = cal_location(row['flight_routes'], cur_time)
        all_flights_locations.append((long, lat))
    return all_flights_locations

def cal_flow_grids(all_locations, num_steps):
    # long_min, long_max = 970000, 1080000
    # lat_min, lat_max = 260000, 340000
    # lat_min, lat_max = 21.5, 34.14
    # long_min, long_max = 87.22, 109.52
    # # big area
    # lat_min, lat_max = 20, 45
    # long_min, long_max = 90, 120
    # big area
    lat_min, lat_max = 22.0, 37.0
    long_min, long_max = 107.0, 122.0
    lat_step = (lat_max - lat_min) / num_steps
    long_step = (long_max - long_min) / num_steps
    flow_mat = np.zeros(shape=(num_steps, num_steps), dtype=np.int32)
    for location in all_locations:
        if location[0] == -1 or location[1] == -1:
            continue
        long_index = int(np.floor((location[0] - long_min) / long_step))
        lat_index = int(np.floor((location[1] - lat_min) / lat_step))
        if lat_index < num_steps and lat_index >= 0 and long_index < num_steps and long_index >= 0:
            flow_mat[lat_index, long_index] += 1
    return flow_mat

date = '202105'
df_real = pd.read_csv("./FlightsPlan_{}.csv".format(date))
print(df_real.shape)
df_real['len_flight_routes'] = df_real['flight_routes'].apply(lambda x: len(eval(x)))
df_noRoutes = df_real[df_real['len_flight_routes']==0]
print(df_real[df_real['len_flight_routes']==0].shape)
df_real = df_real[df_real['len_flight_routes']!=0]
print(df_real.shape)

for index, row in df_real.iterrows():
    flight_routes = row['flight_routes']
    if index % 10000 == 0:
        print(index)
    try:
       np.max([t[0] for t in eval(flight_routes) if t[0] is not None])
    except:
        print(flight_routes)
        embed()


df_real['max_route_time'] = df_real['flight_routes'].apply(lambda x: np.max([t[0] for t in eval(x) if t[0] is not None]))
df_real['min_route_time'] = df_real['flight_routes'].apply(lambda x: np.min([t[0] for t in eval(x) if t[0] is not None]))


print(df_real.shape,
      df_real[df_real['departure_airport'].isna()].shape,
      df_real[df_real['departure_time'].isna()].shape,
      df_real[df_real['arrival_airport'].isna()].shape,
      df_real[df_real['arrival_time'].isna()].shape,
      df_real[df_real['is_miss_info']].shape,
      df_real[(df_real['departure_airport'].isna()) | (df_real['departure_time'].isna()) | (df_real['arrival_airport'].isna()) | (df_real['arrival_time'].isna())].shape)


all_locations_cur_time = []
start_time = int(time.mktime(time.strptime('202106010000', "%Y%m%d%H%M")))
end_time = int(time.mktime(time.strptime('202107010000', "%Y%m%d%H%M")))
for cur_t in range(start_time, end_time, 600):
    print("Processing TimeStample is {0}, Time is {1}".format(str(cur_t), str(time.localtime(cur_t))))
    all_flights_locations = calculate_locations_all(df_real, cur_t)
    all_locations_cur_time.append((cur_t, all_flights_locations))

df_locations = pd.DataFrame(all_locations_cur_time, columns=['cur_time', 'all_locations'])
df_locations.to_csv('locations_plan_2106_small.csv')
#
df_loc = pd.read_csv('locations_plan_2106_small.csv')
num_grids = 10
flow_cube = np.zeros(shape=(df_loc.shape[0], num_grids, num_grids))
print(np.sum(flow_cube), flow_cube.shape)
for index, row in df_loc.iterrows():
    grids_flow = cal_flow_grids(eval(row['all_locations']), num_grids)
    flow_cube[index, :, :] = grids_flow


print(np.sum(flow_cube), flow_cube.shape)
np.save('flow_plan_2106_small.npy', flow_cube)
