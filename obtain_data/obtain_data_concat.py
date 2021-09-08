import numpy as np


flow_real_8 = np.load('./flow_real_2105_small.npy')   # 31 days
flow_real_9 = np.load('./flow_real_2106_small.npy')  # 30 days
# flow_real_9 = flow_real_8[:30*24*6, :, :]  # Fake Data of 9th month
flow_real_all = np.concatenate([flow_real_8, flow_real_9], axis=0)
np.save('./flow_real_small_2021.npy', flow_real_all)
print("FlowReal Shape: ", flow_real_all.shape)

flow_plan_8 = np.load('./flow_plan_2105_small.npy')
flow_plan_9 = np.load('./flow_plan_2106_small.npy')
flow_plan_all = np.concatenate([flow_plan_8, flow_plan_9], axis=0)
np.save('./flow_plan_small_2021.npy', flow_plan_all)
print("FlowPlan Shape", flow_plan_all.shape)

# # weather = np.load('./weather_feature.npy')
# weather = np.random.random(size=((31+30)*24*6, 10, 10))  # Fake Weather
# np.save('./weather_feature.npy', weather)  # save fake data . You should cal real data, and needn't save.
# print("Weather Shape: ", weather.shape)
