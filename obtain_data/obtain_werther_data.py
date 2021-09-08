import os
import numpy as np
import time


def cal_feature_single(weather_arr, num_steps):
    # weather_arr  is an array shape of ? * 3 (lat, long, feature)
    # output: weather info shape(num_grids * num_grids)

    # lat_min, lat_max = 20.0, 45.0
    # long_min, long_max = 90.0, 120.0

    lat_min, lat_max = 22.0, 37.0
    long_min, long_max = 107.0, 122.0

    lat_step = (lat_max - lat_min) / num_steps
    long_step = (long_max - long_min) / num_steps

    weather_arr = weather_arr.astype(np.float)
    weather_arr[:, 0] = np.floor((weather_arr[:, 0] - lat_min) / lat_step)
    weather_arr[:, 1] = np.floor((weather_arr[:, 1] - long_min) / long_step)

    weather_arr = weather_arr[weather_arr[:, 0] >= 0]
    weather_arr = weather_arr[weather_arr[:, 0] < num_steps]
    weather_arr = weather_arr[weather_arr[:, 1] >= 0]
    weather_arr = weather_arr[weather_arr[:, 1] < num_steps]

    weather_sum = np.zeros(shape=[num_grids, num_grids])
    weather_count = np.zeros(shape=[num_grids, num_grids])
    for r in range(weather_arr.shape[0]):
        index_i, index_j, feature = weather_arr[r, :]
        index_i, index_j = int(index_i), int(index_j)
        weather_count[index_i][index_j] += 1
        weather_sum[index_i][index_j] += feature

    weather_count[weather_count == 0] = 1
    weather_arr_grid = weather_sum / weather_count
    weather_arr_grid[weather_arr_grid < 6.0] = 6.0
    return weather_arr_grid


# 201909
start_time = int(time.mktime(time.strptime('202105010000', "%Y%m%d%H%M")))
end_time = int(time.mktime(time.strptime('202107010000', "%Y%m%d%H%M")))
file_path = '/media/data/shenzhiqi/Coding/data_weather/data_wether_202105_06_arrays/'



st = time.time()
num_grids = 10*20
feature_all = np.zeros(shape=[(end_time-start_time) // 600 + 1, num_grids, num_grids])
for index, cur_t in enumerate(range(start_time, end_time, 600)):
    cur_time = time.strftime("%Y%m%d%H%M%S", time.localtime(cur_t))
    if os.path.exists(file_path + cur_time + '.npy'):
        weather_arr = np.load(file_path + cur_time + '.npy')
        feature = cal_feature_single(weather_arr, num_grids)
    else:
        feature = np.zeros(shape=[num_grids, num_grids], dtype=np.float)
    feature_all[index, :, :] = feature
    print("Processing date is {0}, and cost time is {1}".format(time.strftime("%Y%m%d%H%M%S", time.localtime(cur_t)), time.time() - st))

np.save('weather_feature_small_20210506.npy', feature_all)
