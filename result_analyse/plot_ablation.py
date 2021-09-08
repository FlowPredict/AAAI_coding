import numpy as np
import matplotlib.pyplot as plt
#
flow_label = np.load('flow_label.npy')
flow_predict_distill = np.load('flow_distill.npy')
flow_predict_student = np.load('flow_student.npy')
flow_predict_student_no_weather = np.load('flow_student_no_weather.npy')


# MSE 时刻平均(最后一维是时间)
mse_distill = np.mean(np.mean(np.mean(np.square(flow_label - flow_predict_distill), axis=0), axis=0), axis=0)
mse_student = np.mean(np.mean(np.mean(np.square(flow_label - flow_predict_student), axis=0), axis=0), axis=0)
mse_student_no_weather = np.mean(np.mean(np.mean(np.square(flow_label - flow_predict_student_no_weather), axis=0), axis=0), axis=0)

# RMSE
rmse_distill = np.sqrt(mse_distill)
rmse_student = np.sqrt(mse_student)
rmse_student_no_weather = np.sqrt(mse_student_no_weather)

# MAE
mae_distill = np.mean(np.mean(np.mean(np.abs(flow_label - flow_predict_distill), axis=0), axis=0), axis=0)
mae_student = np.mean(np.mean(np.mean(np.abs(flow_label - flow_predict_student), axis=0), axis=0), axis=0)
mae_student_no_weather = np.mean(np.mean(np.mean(np.abs(flow_label - flow_predict_student_no_weather), axis=0), axis=0), axis=0)

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

ablation_option = 'teacher'#'weather'
index = 'MAPE' #'RMSE'

f = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 22
}
f_legend = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 17
}
bar_width = 0.2  # 条形宽度
T = ["10", "20", "30", "40", "50", "60"]
if ablation_option == 'teacher':
    # teacher
    index_distill = np.arange(len(rmse_distill))  # 的横坐标
    index_student = index_distill + bar_width  # 横坐标
    if index == 'RMSE':
        plt.bar(index_distill, height=rmse_distill, width=bar_width, color='b', label='own')
        plt.bar(index_student, height=rmse_student, width=bar_width, color='c', label='student')
        plt.ylabel('RMSE', fontdict=f)  # 纵坐标轴标题
        plt.ylim(2.45, 2.58)
    elif index == 'MAE':
        plt.bar(index_distill, height=mae_distill, width=bar_width, color='b', label='own')
        plt.bar(index_student, height=mae_student, width=bar_width, color='c', label='student')
        plt.ylabel('MAE', fontdict=f)  # 纵坐标轴标题
        plt.ylim(1.72, 1.83)
    elif index == 'MAPE':
        plt.bar(index_distill, height=mape_distill, width=bar_width, color='b', label='own')
        plt.bar(index_student, height=mape_student, width=bar_width, color='c', label='student')
        plt.ylabel('MAPE', fontdict=f)  # 纵坐标轴标题
        plt.ylim(0.34,0.39)
    plt.legend(['ST-KDN', 'ST-KDN-NT'], prop=f_legend,loc=2)  # 显示图例
    # plt.title('Effect of Teacher Guidance', fontdict=f1)  # 图形标题

elif ablation_option == 'weather':
    index_student = np.arange(len(rmse_student))  # 的横坐标
    index_student_no_weather = index_student + bar_width  # 横坐标
    if index == 'RMSE':
        plt.bar(index_student, height=rmse_student, width=bar_width, color='b', label='student')
        plt.bar(index_student_no_weather, height=rmse_student_no_weather, width=bar_width, color='c', label='NoWeather')
        plt.ylabel('RMSE', fontdict=f)  # 纵坐标轴标题
        plt.ylim(2.45, 2.6)
    elif index == 'MAE':
        plt.bar(index_student, height=mae_student, width=bar_width, color='b', label='student')
        plt.bar(index_student_no_weather, height=mae_student_no_weather, width=bar_width, color='c', label='NoWeather')
        plt.ylabel('MAE', fontdict=f)  # 纵坐标轴标题
        plt.ylim(1.76,1.82)
    elif index == 'MAPE':
        plt.bar(index_student, height=mape_student, width=bar_width, color='b', label='student')
        plt.bar(index_student_no_weather, height=mape_student_no_weather, width=bar_width, color='c', label='NoWeather')
        plt.ylabel('MAPE', fontdict=f)  # 纵坐标轴标题
        plt.ylim(0.35,0.40)
    plt.legend(['ST-KDN-NT','ST-KDN-NTW'],prop=f_legend,loc=2) # 显示图例
    # plt.title('Effect of Weather Feature Modeling',fontdict=f1)  # 图形标题

plt.subplots_adjust(left=0.2,bottom=0.2)
plt.xticks(index_student + bar_width / 2, T, fontproperties='Times New Roman',
               size=22)  #
plt.yticks(fontproperties='Times New Roman', size=22)
plt.xlabel('Time/min', fontdict=f)


plt.show()