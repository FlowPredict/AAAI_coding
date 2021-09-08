import numpy as np
import matplotlib.pyplot as plt
#
# 查看数据的缺失情况
real_data_path = "../obtain_data/flow_real_big.npy"
plan_data_path = "../obtain_data/flow_plan_big.npy"
weather_data_path = "../obtain_data/weather_feature_big.npy"

flow_data_raw = np.load(real_data_path)
all_flows = list(np.sum(flow_data_raw, axis=(1, 2)))
print(len(all_flows))
print(np.min(all_flows))
print(np.mean(all_flows))
# 真实流量数据中没有0
plt.figure("flows")
plt.plot(all_flows)
plt.show()

plan_data_raw = np.load(plan_data_path)
all_plan = list(np.sum(plan_data_raw, axis=(1, 2)))
print(len(all_plan))
print(np.min(all_plan))
print(np.mean(all_plan))
for index, p in enumerate(all_plan):
    if p == 0.0:
        print(index, end=',')
print()
plt.figure("plans")
plt.plot(all_flows)
plt.show()



weather_data_raw = np.load(weather_data_path)
all_weather = list(np.mean(weather_data_raw, axis=(1, 2)))
print(len(all_weather))
print(np.min(all_weather))
for index, w in enumerate(all_weather):
    if w == 3.0:
        # 8163 之后的都是6分钟间隔的, 不做处理
        if index <= 8163:
            print(index, end=',')
            print(np.mean(weather_data_raw[index-1, :, :]), np.mean(weather_data_raw[index, :, :]), np.mean(weather_data_raw[index+1, :, :]))


print()
print(np.min(weather_data_raw))

plt.figure()
plt.plot(all_weather)
plt.show()