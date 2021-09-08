#
# 天气数据插值
import numpy as np
import matplotlib.pyplot as plt
weather_data_path = "../obtain_data/weather_feature_small_20210506.npy"

weather_data_raw = np.load(weather_data_path)
all_weather = list(np.mean(weather_data_raw, axis=(1, 2)))
print(len(all_weather))
plt.figure("before")
plt.plot(all_weather)
plt.show()


# step 1: 找到 缺失的index
miss_indexes = []
for index, w in enumerate(all_weather):
    if w == 0.0:  # 处理数据的时候，将缺失数据置为3了
        # if index >= 8163:
        #     # 8163 之后的都是6分钟间隔的, 不做处理
        #     continue
        miss_indexes.append(index)

sub_lianxu_indexes = [miss_indexes[0]]   # 用来存储连续的一段index
for idx in miss_indexes[1:]:  # 第一个已经添加到sub_lianxu_indexes
    if idx == sub_lianxu_indexes[-1] + 1:
        sub_lianxu_indexes.append(idx)
    else:
        # 找到了一段连续的indexes，进行赋值处理
        print("Processing: ", sub_lianxu_indexes)
        idx_start = sub_lianxu_indexes[0] - 1
        idx_end = sub_lianxu_indexes[-1] + 1
        idx_dis = idx_end - idx_start
        for sub_idx in sub_lianxu_indexes:
            weather_data_raw[sub_idx, :, :] = (sub_idx - idx_start) / idx_dis * weather_data_raw[idx_start, :, :] + \
                                              (idx_end - sub_idx) / idx_dis * weather_data_raw[idx_end, :, :]
        sub_lianxu_indexes = [idx]

all_weather_after = list(np.mean(weather_data_raw, axis=(1, 2)))
plt.figure("after")
plt.plot(all_weather_after)
plt.show()
print(weather_data_raw.shape)
np.save('weather_feature_small_interpplation_2021.npy', weather_data_raw)
