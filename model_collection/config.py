# -*- coding: utf-8 -*-
import os
from easydict import EasyDict

cfg = EasyDict()

cfg.train = EasyDict()
cfg.train.model_name = "STGCNModel"
cfg.train.model_name_distill = "STGCNModel"  # TransformerDistillModel, CnnDistillModel
cfg.train.batch_size = 64
cfg.train.grid_size = 10  # grids size of train data
cfg.train.grid_size_weather = 200  # grids size of weather data
cfg.train.input_P = 18  # history P times
cfg.train.output_Q = 6  # predict Q times
cfg.train.init_lr = 0.01  # init learning rate
cfg.train.epochs = 50  # train epochs

use_data_set = '2021'
if use_data_set == '2021':
    cfg.train.real_data_path = "../obtain_data/flow_real_small_2021.npy"
    cfg.train.plan_data_path = "../obtain_data/flow_plan_small_2021.npy"
    cfg.train.weather_data_path = "../obtain_data/weather_feature_small_interpplation_2021.npy"
    cfg.train.res_save_path = "./res_2021"