import time
import numpy as np
import os

def read_file(file_name):
    fp = open(file_name, 'r')
    lines = fp.readlines()
    lines = [line[:-1].split(' ') for line in lines]  # -1 : delet '\n'
    return lines


start_time = int(time.mktime(time.strptime('202106010000', "%Y%m%d%H%M")))
end_time = int(time.mktime(time.strptime('202107010000', "%Y%m%d%H%M")))


file_path = '/media/data/data/radarecho/202106'
file_save_path = '/media/data/shenzhiqi/Coding/data_weather/data_wether_202105_06_arrays/'

st = time.time()

for index, cur_t in enumerate(range(start_time, end_time, 600)):
    cur_time = time.strftime("%Y%m%d%H%M%S", time.localtime(cur_t))
    # date = 'radart' + cur_time[:8]
    date = cur_time[:8]
    file_time = time.strftime("%Y%m%d%H%M%S", time.localtime(cur_t - 8 * 3600))
    date_file, hms_file = file_time[:8], file_time[8:]
    file_name = 'Z_RADA_C_BABJ_{0}_P_ACHN.QREF000.{1}.{2}.latlon.dat'.format(str(file_time), str(date_file),
                                                                             str(hms_file))
    if os.path.exists(os.path.join(file_path, date, file_name)):
        lines = read_file(os.path.join(file_path, date, file_name))
        arr = np.array(lines)
        np.save(file_save_path + cur_time + '.npy', arr)
    else:
        print("False")
    print("Processing date is {0}, and cost time is {1}".format(cur_time, time.time() - st))
