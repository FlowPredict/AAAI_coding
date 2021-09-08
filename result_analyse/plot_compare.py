import numpy as np
import matplotlib.pyplot as plt
#
flow_label = np.load('flow_label.npy')
flow_predict_distill = np.load('flow_distill.npy')
flow_predict_student = np.load('flow_student.npy')
flow_predict_student_no_weather = np.load('flow_student_no_weather.npy')
flow_GMAN = np.reshape(np.transpose(np.load('flow_GMAN.npy'), axes=[0, 2, 1]), newshape=[-1, 10, 10, 6])
flow_ASTGCN = np.reshape(np.transpose(np.load('flow_ASTGCN.npy'), axes=[0, 2, 1]), newshape=[-1, 10, 10, 6])
flow_AGCRN = np.reshape(np.transpose(np.load('flow_AGCRN.npy'), axes=[0, 2, 1, 3]), newshape=[-1, 10, 10, 6])
flow_SVR = np.reshape(np.load('flow_SVR.npy'), newshape=[-1, 10, 10, 6])[:384, :, :, :]

# MSE 时刻平均(最后一维是时间)
mse_distill = np.mean(np.mean(np.mean(np.square(flow_label - flow_predict_distill), axis=0), axis=0), axis=0)
mse_student = np.mean(np.mean(np.mean(np.square(flow_label - flow_predict_student), axis=0), axis=0), axis=0)
mse_student_no_weather = np.mean(np.mean(np.mean(np.square(flow_label - flow_predict_student_no_weather), axis=0), axis=0), axis=0)

mse_GMAN = np.mean(np.mean(np.mean(np.square(flow_label - flow_GMAN), axis=0), axis=0), axis=0)
mse_ASTGCN = np.mean(np.mean(np.mean(np.square(flow_label - flow_ASTGCN), axis=0), axis=0), axis=0)
mse_AGCRN = np.mean(np.mean(np.mean(np.square(flow_label - flow_AGCRN), axis=0), axis=0), axis=0)
mse_SVR = np.mean(np.mean(np.mean(np.square(flow_label - flow_SVR), axis=0), axis=0), axis=0)

# RMSE
rmse_distill = np.sqrt(mse_distill)
rmse_student = np.sqrt(mse_student)
rmse_student_no_weather = np.sqrt(mse_student_no_weather)
rmse_GMAN = np.sqrt(mse_GMAN)
rmse_ASTGCN = np.sqrt(mse_ASTGCN)
rmse_AGCRN = np.sqrt(mse_AGCRN)
rmse_SVR = np.sqrt(mse_SVR)

# MAE
mae_distill = np.mean(np.mean(np.mean(np.abs(flow_label - flow_predict_distill), axis=0), axis=0), axis=0)
mae_student = np.mean(np.mean(np.mean(np.abs(flow_label - flow_predict_student), axis=0), axis=0), axis=0)
mae_student_no_weather = np.mean(np.mean(np.mean(np.abs(flow_label - flow_predict_student_no_weather), axis=0), axis=0), axis=0)
mae_GMAN = np.mean(np.mean(np.mean(np.abs(flow_label - flow_GMAN), axis=0), axis=0), axis=0)
mae_ASTGCN = np.mean(np.mean(np.mean(np.abs(flow_label - flow_ASTGCN), axis=0), axis=0), axis=0)
mae_AGCRN = np.mean(np.mean(np.mean(np.abs(flow_label - flow_AGCRN), axis=0), axis=0), axis=0)
mae_SVR = np.mean(np.mean(np.mean(np.abs(flow_label - flow_SVR), axis=0), axis=0), axis=0)

# MAPE
def cal_mape(flow_label, flow_predict):
    diff = np.abs(flow_label - flow_predict)
    diff_ratio = diff / flow_label
    # diff_ratio[diff_ratio == np.math.nan] = 0.0
    diff_ratio[diff_ratio == np.math.inf] = 0.0
    return np.mean(np.mean(np.mean(diff_ratio, axis=0), axis=0), axis=0)

mape_distill = cal_mape(flow_label, flow_predict_distill)
mape_student = cal_mape(flow_label, flow_predict_student)
mape_student_no_weather = cal_mape(flow_label, flow_predict_student_no_weather)
mape_GMAN = cal_mape(flow_label, flow_GMAN)
mape_ASTGCN = cal_mape(flow_label, flow_ASTGCN)
mape_AGCRN = cal_mape(flow_label, flow_AGCRN)
mape_SVR = cal_mape(flow_label, flow_SVR)

print("MAE OF distill:",mae_distill)
print("MAE OF GMAN:",mae_GMAN)
print("MAE OF ASTGCN:",mae_ASTGCN)
print("MAE OF AGCRN:",mae_AGCRN)
print("MAE OF STGCN:",mae_student_no_weather)
print("MAE OF SVR:",mae_SVR)

print("RMSE OF distill:",rmse_distill)
print("RMSE OF GMAN:",rmse_GMAN)
print("RMSE OF ASTGCN:",rmse_ASTGCN)
print("RMSE OF AGCRN:",rmse_AGCRN)
print("RMSE OF STGCN:",rmse_student_no_weather)
print("RMSE OF SVR:",rmse_SVR)

print("MAPE OF distill:",mape_distill)
print("MAPE OF GMAN:",mape_GMAN)
print("MAPE OF ASTGCN:",mape_ASTGCN)
print("MAPE OF AGCRN:",mape_AGCRN)
print("MAPE OF STGCN:",mape_student_no_weather)
print("MAPE OF SVR:",mape_SVR)
