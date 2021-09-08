# -*- coding: utf-8 -*-
import numpy as np
import random
from model_collection.config import cfg

class DataLoader(object):
    def __init__(self, train_days=30, eval_days=3, test_days=10):
        self.flow_data_raw = np.transpose(np.load(cfg.train.real_data_path), axes=[1, 2, 0])
        # self.flow_data_raw = self.flow_data_raw / np.max(self.flow_data_raw)
        self.flow_data_train = self.flow_data_raw[:, :, :train_days*24*6]
        self.flow_data_eval = self.flow_data_raw[:, :, train_days*24*6:(train_days+eval_days)*24*6]
        self.flow_data_test = self.flow_data_raw[:, :, (train_days+eval_days)*24*6:(train_days+eval_days+test_days)*24*6]

        self.plan_data_raw = np.transpose(np.load(cfg.train.plan_data_path), axes=[1, 2, 0])
        # self.plan_data_raw = self.plan_data_raw / np.max(self.plan_data_raw)
        self.plan_data_train = self.plan_data_raw[:, :, :train_days*24*6]
        self.plan_data_eval = self.plan_data_raw[:, :, train_days*24*6:(train_days+eval_days)*24*6]
        self.plan_data_test = self.plan_data_raw[:, :, (train_days+eval_days)*24*6:(train_days+train_days+test_days)*24*6]

        self.weather_data_raw = np.transpose(np.load(cfg.train.weather_data_path), axes=[1, 2, 0])
        # self.weather_data_raw = self.weather_data_raw / np.max(self.weather_data_raw)
        self.weather_data_train = self.weather_data_raw[:, :, :train_days*24*6]
        self.weather_data_eval = self.weather_data_raw[:, :, train_days*24*6:(train_days+eval_days)*24*6]
        self.weather_data_test = self.weather_data_raw[:, :, (train_days+eval_days)*24*6:(train_days+train_days+test_days)*24*6]

        self.batch_size = cfg.train.batch_size
        self.input_P = cfg.train.input_P
        self.output_Q = cfg.train.output_Q
        self.grid_size = cfg.train.grid_size
        self.grid_size_weather = cfg.train.grid_size_weather
        self.train_samples = self.flow_data_train.shape[-1] - self.input_P - self.output_Q + 1
        self.eval_samples = self.flow_data_eval.shape[-1] - self.input_P - self.output_Q + 1
        self.test_samples = self.flow_data_test.shape[-1] - self.input_P - self.output_Q + 1
        self.train_batch_num_per_epoch = self.train_samples // self.batch_size
        self.eval_batch_num_per_epoch = self.eval_samples // self.batch_size
        self.test_batch_num_per_epoch = self.test_samples // self.batch_size

    def create_generator(self, mode='train'):
        # mode: train, eval or test
        if mode == 'train':
            samples = self.train_samples
            sample_indexes = list(range(samples))
            random.shuffle(sample_indexes)
            flow_data = self.flow_data_train
            plan_data = self.plan_data_train
            weather_data = self.weather_data_train
        elif mode == 'eval':
            samples = self.eval_samples
            sample_indexes = list(range(samples))
            flow_data = self.flow_data_eval
            plan_data = self.plan_data_eval
            weather_data = self.weather_data_eval
        elif mode == 'test':
            samples = self.test_samples
            sample_indexes = list(range(samples))
            flow_data = self.flow_data_test
            plan_data = self.plan_data_test
            weather_data = self.weather_data_test
        else:
            raise Exception("Invalid Generator Mode!!!")

        batch_flow_data = np.zeros(shape=(self.batch_size, self.grid_size, self.grid_size, self.input_P + self.output_Q))
        batch_plan_data = np.zeros(shape=(self.batch_size, self.grid_size, self.grid_size, self.input_P + self.output_Q))
        batch_weather_data = np.zeros(shape=(self.batch_size, self.grid_size_weather, self.grid_size_weather, self.input_P + self.output_Q))
        cur_index = 0
        while True:
            if cur_index + self.batch_size <= samples:
                for idx, sample_idx in enumerate(sample_indexes[cur_index: cur_index+self.batch_size]):
                    batch_flow_data[idx, :, :, :] = flow_data[:, :, sample_idx:sample_idx+self.input_P+self.output_Q]
                    batch_plan_data[idx, :, :, :] = plan_data[:, :, sample_idx:sample_idx+self.input_P+self.output_Q]
                    batch_weather_data[idx, :, :, :] = weather_data[:, :, sample_idx:sample_idx + self.input_P + self.output_Q]
                cur_index += self.batch_size
                yield batch_flow_data[:, :, :, :self.input_P], batch_flow_data[:, :, :, self.input_P:], \
                      batch_plan_data[:, :, :, :self.input_P], batch_plan_data[:, :, :, self.input_P:], \
                      batch_weather_data
            else:
                raise StopIteration

if __name__ == "__main__":
    test = DataLoader()
    g = test.create_generator(mode='test')
    print(test.test_batch_num_per_epoch)
    for i in range(test.test_batch_num_per_epoch):
        batch_flow_P, batch_flow_Q, batch_plan_P, batch_plan_Q, batch_weather = next(g)
        print(batch_flow_P.shape, batch_flow_Q.shape, batch_plan_P.shape, batch_plan_Q.shape, batch_weather.shape)