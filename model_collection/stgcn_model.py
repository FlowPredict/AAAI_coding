# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from model_collection.config import cfg
from model_collection.layers import st_conv_block, output_layer, fully_con_layer, layer_norm
from model_collection.math_graph import scaled_laplacian, cheb_poly_approx
from model_collection.ops import bn, conv2d

class STGCNModel(object):
    """Implement STGCN model"""
    def __init__(self):
        self.model_name = "STGCNModel"
        self.blocks = [[1, 32, 64], [64, 32, 128]]  # channel configs of st_conv blocks.
        self.Ks = 3  # kernel size of spatial convolution.
        self.Kt = 3  # kernel size of temporal convolution.
        self.weights_file = '../obtain_data/WeightsGraphDis.npy'

    def create_placeholders(self):
        self.input_history_flow_ph = tf.placeholder(dtype=tf.float32, shape=(None, cfg.train.grid_size, cfg.train.grid_size, cfg.train.input_P))
        self.ouput_feature_flow_ph = tf.placeholder(dtype=tf.float32, shape=(None, cfg.train.grid_size, cfg.train.grid_size, cfg.train.output_Q))
        self.input_history_plan_ph = tf.placeholder(dtype=tf.float32, shape=(None, cfg.train.grid_size, cfg.train.grid_size, cfg.train.input_P))
        self.ouput_feature_plan_ph = tf.placeholder(dtype=tf.float32, shape=(None, cfg.train.grid_size, cfg.train.grid_size, cfg.train.output_Q))
        self.input_weather_ph = tf.placeholder(dtype=tf.float32, shape=(None, 200, 200, cfg.train.input_P + cfg.train.output_Q))
        self.is_training_student_ph = tf.placeholder(dtype=tf.bool, shape=())
        self.is_training_teacher_ph = tf.placeholder(dtype=tf.bool, shape=())
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
        return self.input_history_plan_ph, self.ouput_feature_plan_ph, self.input_history_flow_ph, self.ouput_feature_flow_ph,  self.input_weather_ph, \
               self.keep_prob, self.is_training_teacher_ph, self.is_training_student_ph

    def create_weather_extract_module(self, input_weather, is_training):
        with tf.variable_scope("weather_extract_module"):
            # x = res_block(input_weather, is_training=is_training, scope_name='res_block_1')
            x = tf.nn.relu(bn(conv2d(input_weather, 16, 3, 3, 1, 1, scope_name='conv1', padding='SAME'), is_training=is_training, scope='bn1'))
            x = tf.nn.max_pool(input_weather, ksize=2, strides=2, padding="VALID", name='pool_1')
            # x = res_block(x, is_training=is_training, scope_name='res_block_2')
            x = tf.nn.relu(bn(conv2d(x, 16, 3, 3, 1, 1, scope_name='conv2', padding='SAME'), is_training=is_training,
                              scope='bn2'))
            x = tf.nn.max_pool(x, ksize=2, strides=2, padding="VALID", name='pool_2')
            # x = res_block(x, is_training=is_training, scope_name='res_block_3')
            x = tf.nn.relu(bn(conv2d(x, 8, 3, 3, 1, 1, scope_name='conv3', padding='SAME'), is_training=is_training,
                              scope='bn3'))
            x = tf.nn.max_pool(x, ksize=2, strides=2, padding="VALID", name='pool_3')
            x = tf.nn.relu(bn(conv2d(x, 8, 3, 3, 2, 2, scope_name='conv4', padding='VALID'), is_training=is_training, scope='bn4'))
            x = tf.nn.tanh(bn(conv2d(x, 8, 3, 3, 1, 1, scope_name='conv5', padding='VALID'), is_training=is_training, scope='bn5'))
            return x

    def create_weather_extract_stgcn_module(self, input_weather, is_training, keep_prob):
        with tf.variable_scope("weather_extract_module"):
            # 做图卷积的话，需要先把天气变形为 B x 10 x 10 x PQ x 400 （10x10个节点， 每个节点的每个时刻有400个特征）
            _, _, _, PQ = input_weather.get_shape().as_list()  # [B,10,10,PQ]
            input_weather = tf.reshape(input_weather, [-1, 10, 20, 10, 20, PQ])
            input_weather = tf.transpose(input_weather, perm=[0, 1, 3, 5, 2, 4])  #  B x 10 x 10 x PQ x 400
            input_weather = tf.reshape(input_weather, [-1, 100, PQ, 400])   # B x 100 x PQ x 400
            x = tf.transpose(input_weather, perm=[0, 2, 1, 3])  # [B,PQ,100,D]
            # ST-Block
            x = st_conv_block(x, self.Ks, self.Kt, [400, 32, 64], 'weather_extract', keep_prob, act_func='relu')  # B, PQ-Kt+1, 100, D
            x = tf.transpose(x, perm=[0, 2, 3, 1])  # [B,10*10,D,PQ-Kt+1]
            x = tf.nn.relu(bn(conv2d(x, 14, k_h=1, k_w=1, d_h=1, d_w=1, scope_name='conv1', padding="VALID"), is_training=is_training, scope='bn1'))  # [B,10*10,D, 16]
            x = tf.transpose(x, perm=[0, 3, 1, 2])  # [B,16,10*10, D]
        return x

    def create_stgcn_teacher(self, input_history, keep_prob):
        with tf.variable_scope(self.model_name + "_teacher"):
            _, NumGrids, _, P = input_history.get_shape().as_list()  #[B,10,10,P]
            x = tf.reshape(input_history, shape=(-1, NumGrids*NumGrids, P, 1))  #[B,10*10,P,1]
            x = tf.transpose(x, perm=[0, 2, 1, 3])  #[B,P,10*10,1]
            # ST-Block
            for i, channels in enumerate(self.blocks):
                x = st_conv_block(x, self.Ks, self.Kt, channels, i, keep_prob, act_func='GLU')
                if i == 0:
                    x_distill = x  # [B,P,10*10,64]
            # Output Layer
            y = output_layer(x, self.Kt, cfg.train.output_Q, 'output_layer')  #[B,Q,10*10,?]
            y = y[:, :, :, 0]
            y = tf.transpose(y, perm=[0, 2, 1])  #[B,10*10,Q]
            output = tf.reshape(y, shape=(-1, NumGrids, NumGrids, cfg.train.output_Q))  #[B,10,10,Q]
        return x_distill, output

    def create_stgcn_student(self, input_history, input_weather, is_training, keep_prob, use_weather=True):
        with tf.variable_scope(self.model_name + '_student'):
            _, NumGrids, _, P = input_history.get_shape().as_list()
            x = tf.reshape(input_history, shape=(-1, NumGrids*NumGrids, P, 1))  #[B,10*10,P,1]
            x = tf.transpose(x, perm=[0, 2, 1, 3])  #[B,P,10*10,1]
            # ST-Block
            for i, channels in enumerate(self.blocks):
                x = st_conv_block(x, self.Ks, self.Kt, channels, i, keep_prob, act_func='relu')
                if i == 0:
                    x_distill = x
                    if use_weather:
                        # 卷积
                        # weather_feature = self.create_weather_extract_module(input_weather, is_training)  # [B,P,10*10,64]
                        # _, time_steps, _, _ = x.get_shape().as_list()
                        # weather_feature = tf.reshape(weather_feature, shape=(-1, NumGrids * NumGrids, 8, 1))  # [B,10*10,16,1]
                        # weather_feature = tf.transpose(weather_feature, perm=[0, 3, 1, 2])  # [B,1,10*10,16]
                        # weather_feature = tf.tile(weather_feature, multiples=[1, time_steps, 1, 1])
                        # STGCN
                        # [B,16,10*10,D]
                        weather_feature = self.create_weather_extract_stgcn_module(input_weather, is_training, keep_prob)
                        # Concat
                        x = tf.concat([x, weather_feature], axis=-1)
                        # x = x + weather_feature
                        x = conv2d(x, output_dim=64,  k_h=1, k_w=1, d_h=1, d_w=1, scope_name='conv_concat', name='weight_decay')

            # Output Layer
            y = output_layer(x, self.Kt, cfg.train.output_Q, 'output_layer')  #[B,Q,10*10,1]
            y = y[:, :, :, 0]  #[B,Q,10*10]
            y = tf.transpose(y, perm=[0, 2, 1])  #[B,10*10,Q]
            output = tf.reshape(y, shape=(-1, NumGrids, NumGrids, cfg.train.output_Q))  #[B,10,10,Q]
        return x_distill, output

    def create_stgcn_distill(self, input_history, input_weather, is_training, keep_prob, use_weather=True):
        with tf.variable_scope(self.model_name + '_student'):
            _, NumGrids, _, P = input_history.get_shape().as_list()
            x = tf.reshape(input_history, shape=(-1, NumGrids * NumGrids, P, 1))  # [B,10*10,P,1]
            x = tf.transpose(x, perm=[0, 2, 1, 3])  # [B,P,10*10,1]
            # ST-Block
            for i, channels in enumerate(self.blocks):
                x = st_conv_block(x, self.Ks, self.Kt, channels, i, keep_prob, act_func='relu')
                if i == 0:
                    x_distill = x
                    output_dim = x_distill.get_shape().as_list()[-1]
                    x_distill = conv2d(x_distill, output_dim=output_dim,k_h=1,k_w=1,d_h=1,d_w=1, scope_name='conv_distill', name='weight_decay_distill')
                    if use_weather:
                        # 卷积
                        # weather_feature = self.create_weather_extract_module(input_weather, is_training)  # [B,P,10*10,64]
                        # _, time_steps, _, _ = x.get_shape().as_list()
                        # weather_feature = tf.reshape(weather_feature, shape=(-1, NumGrids * NumGrids, 8, 1))  # [B,10*10,16,1]
                        # weather_feature = tf.transpose(weather_feature, perm=[0, 3, 1, 2])  # [B,1,10*10,16]
                        # weather_feature = tf.tile(weather_feature, multiples=[1, time_steps, 1, 1])
                        # STGCN
                        # [B,16,10*10,D]
                        weather_feature = self.create_weather_extract_stgcn_module(input_weather, is_training,
                                                                                   keep_prob)
                        # Concat
                        x = tf.concat([x, weather_feature], axis=-1)
                        # x = x + weather_feature
                        x = conv2d(x, output_dim=64, k_h=1, k_w=1, d_h=1, d_w=1, scope_name='conv_concat',
                                   name='weight_decay')

            # Output Layer
            y = output_layer(x, self.Kt, cfg.train.output_Q, 'output_layer')  # [B,Q,10*10,1]
            y = y[:, :, :, 0]  # [B,Q,10*10]
            y = tf.transpose(y, perm=[0, 2, 1])  # [B,10*10,Q]
            output = tf.reshape(y, shape=(-1, NumGrids, NumGrids, cfg.train.output_Q))  # [B,10,10,Q]
        return x_distill, output

    def load_weights_graph(self):
        #
        W = np.load(self.weights_file)
        # Calculate graph kernel
        L = scaled_laplacian(W)
        # Alternative approximation method: 1st approx - first_approx(W, n).
        Lk = cheb_poly_approx(L, self.Ks, cfg.train.grid_size * cfg.train.grid_size)
        tf.add_to_collection(name='graph_kernel', value=tf.cast(tf.constant(Lk), tf.float32))
        return

