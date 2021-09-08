# -*- coding:utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import time
import random
import numpy as np
import tensorflow as tf
from model_collection.config import cfg
from model_collection.stgcn_model import STGCNModel
from model_collection.dataloader import DataLoader

class TrainTask(object):
    """ Training any model here """

    def __init__(self, sess, use_weather=True):
        self.sess = sess
        self.model_name = cfg.train.model_name
        self.model_name_teacher = cfg.train.model_name + '_teacher'
        self.use_weather = use_weather
        if self.use_weather:
            self.model_name_save = self.model_name + '_distill'
        else:
            self.model_name_save = self.model_name + '_distill_no_weather'
        self.train_epochs = cfg.train.epochs
        self.lr_init = cfg.train.init_lr
        self.build_model(use_weather=self.use_weather)
        self.log_dir = os.path.join(cfg.train.res_save_path, "logs")
        self.checkpoint_dir = os.path.join(cfg.train.res_save_path, "checkpoint")
        self.result_loss_dir = os.path.join(cfg.train.res_save_path, "result_loss", self.model_name_save)
        if not os.path.exists(self.result_loss_dir):
            os.makedirs(self.result_loss_dir)
        self.dataloader = DataLoader()
        self.train_batch_num_per_epoch = self.dataloader.train_batch_num_per_epoch
        self.eval_batch_num_per_epoch = self.dataloader.eval_batch_num_per_epoch
        self.test_batch_num_per_epoch = self.dataloader.test_batch_num_per_epoch


    def build_model(self, use_weather=True):
        if self.model_name == "STGCNModel":
            self.model = STGCNModel()
            _, _, self.input_history_flow_ph, self.ouput_feature_flow_ph, self.input_weather_ph, \
            self.keep_prob, self.is_training_teacher_ph, self.is_training_student_ph = self.model.create_placeholders()
            self.model.load_weights_graph()
            self.f_distill_teacher, _ = self.model.create_stgcn_teacher(self.input_history_flow_ph, self.keep_prob)
            self.f_distill_student, self.output_student = self.model.create_stgcn_distill(self.input_history_flow_ph, self.input_weather_ph, self.is_training_student_ph, self.keep_prob, use_weather=use_weather)
            self.loss_distill = tf.reduce_mean(tf.square(self.f_distill_student - self.f_distill_teacher))
            self.loss_predict = tf.reduce_mean(tf.square(self.ouput_feature_flow_ph - self.output_student))
            self.loss = 0.001 * self.loss_distill + self.loss_predict
        else:
            raise("!!!!!!ModelName dosen't Exist!!!!!!")

        self.global_step = tf.Variable(0, trainable=False)
        self.add_global = self.global_step.assign_add(1)
        self.lr = tf.train.exponential_decay(self.lr_init, global_step=self.global_step, decay_steps=100, decay_rate=0.8)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            # 获取学生网络的参数，下面的优化器只优化学生网络
            all_train_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            print(all_train_weights)
            trainabel_weights_student = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.model_name + '_student')
            print(trainabel_weights_student)
            self.optim_student = tf.train.AdamOptimizer(self.lr).minimize(self.loss, var_list=trainabel_weights_student)
            # self.optim_student = tf.train.AdamOptimizer(self.lr).minimize(self.loss_predict, var_list=trainabel_weights_student)
        """ Summary """
        loss_sum_distill = tf.summary.scalar("loss_distill", self.loss_distill)
        loss_sum_predict = tf.summary.scalar("loss_predict", self.loss_predict)
        loss_sum = tf.summary.scalar("loss", self.loss)
        # final summary operations
        self.summary = tf.summary.merge([loss_sum_distill, loss_sum_predict, loss_sum])


    def train(self):
        variables = tf.contrib.framework.get_variables_to_restore()
        print(variables)
        variables_to_resotre = [v for v in variables if v.name.split('/')[0] == self.model_name + "_teacher"]
        print(variables_to_resotre)

        # saver to save model
        self.saver = tf.train.Saver(var_list=variables_to_resotre, max_to_keep=3)
        # summary writer
        self.writer = tf.summary.FileWriter(os.path.join(self.log_dir, self.model_name_save + '_train'), self.sess.graph)
        self.writer_eval = tf.summary.FileWriter(os.path.join(self.log_dir, self.model_name_save + "_eval"), self.sess.graph)
        tf.global_variables_initializer().run()
        # Loading Pretrained Teacher Model
        self.load(self.model_name_teacher)
        self.saver = tf.train.Saver(max_to_keep=3)

        loss_student_train_epoches = []
        loss_distill_train_epoches = []
        loss_predict_train_epoches = []
        loss_student_eval_epoches = []
        loss_distill_eval_epoches = []
        loss_predict_eval_epoches = []

        start_time = time.time()
        for epoch in range(self.train_epochs):
            lr = sess.run(self.lr)
            print("Epoch: {:2d}, learning_rate: {:.6f}".format(epoch, lr))
            loss_student_train = []
            loss_distill_train = []
            loss_predict_train = []
            g = self.dataloader.create_generator(mode='train')
            for step in range(self.train_batch_num_per_epoch):
                # train_step 是总的step数量（每个epoch累加到一起）
                train_step = sess.run(self.global_step)
                sess.run(self.add_global)
                batch_flow_P, batch_flow_Q, _, _, batch_weather = next(g)
                feed_dict = {self.input_history_flow_ph: batch_flow_P,
                             self.ouput_feature_flow_ph: batch_flow_Q,
                             self.input_weather_ph: batch_weather,
                             self.is_training_teacher_ph: False,
                             self.is_training_student_ph: True,
                             self.keep_prob: 1.0}
                _, summary_str, loss, loss_distill, loss_predict = self.sess.run([self.optim_student, self.summary, self.loss, self.loss_distill, self.loss_predict], feed_dict=feed_dict)
                loss_student_train.append(loss)
                loss_distill_train.append(loss_distill)
                loss_predict_train.append(loss_predict)
                self.writer.add_summary(summary_str, train_step)
                print("Epoch: [{:2d}-{:4d}], Step: {:2d}, time: {:4.2f}, loss: {:.6f}, loss_distill: {:.6f}".format(epoch, step, train_step, time.time() - start_time, loss, loss_distill))
            loss_student_train_epoches.append(loss_student_train)
            loss_distill_train_epoches.append(loss_distill_train)
            loss_predict_train_epoches.append(loss_predict_train)
            # eval
            g = self.dataloader.create_generator(mode='eval')
            loss_student_eval = []
            loss_distill_eval = []
            loss_predict_eval = []
            for step in range(self.eval_batch_num_per_epoch):
                batch_flow_P, batch_flow_Q, _, _, batch_weather = next(g)
                feed_dict = {self.input_history_flow_ph: batch_flow_P,
                             self.ouput_feature_flow_ph: batch_flow_Q,
                             self.input_weather_ph: batch_weather,
                             self.is_training_teacher_ph: False,
                             self.is_training_student_ph: False,
                             self.keep_prob: 1.0}

                summary_str_eval, loss, loss_distill, loss_predict = self.sess.run(
                    [self.summary, self.loss, self.loss_distill, self.loss_predict], feed_dict=feed_dict)
                loss_student_eval.append(loss)
                loss_distill_eval.append(loss_distill)
                loss_predict_eval.append(loss_predict)
                self.writer_eval.add_summary(summary_str_eval, step + epoch * self.eval_batch_num_per_epoch)
            loss_student_eval_epoches.append(loss_student_eval)
            loss_distill_eval_epoches.append(loss_distill_eval)
            loss_predict_eval_epoches.append(loss_predict_eval)
            print("Epoch: {:2d}, Eval DataL loss_predict: {:.6f}, loss_distill: {:.6f}".format(epoch, np.mean(loss_predict_eval_epoches), np.mean(loss_distill_eval_epoches)))
            self.save(self.model_name_save)

        np.save(self.result_loss_dir + "/loss_student_train.npy", np.array(loss_student_train_epoches))
        np.save(self.result_loss_dir + "/loss_distill_train.npy", np.array(loss_distill_train_epoches))
        np.save(self.result_loss_dir + "/loss_predict_train.npy", np.array(loss_predict_train_epoches))
        np.save(self.result_loss_dir + "/loss_student_eval.npy", np.array(loss_student_eval_epoches))
        np.save(self.result_loss_dir + "/loss_distill_eval.npy", np.array(loss_distill_eval_epoches))
        np.save(self.result_loss_dir + "/loss_predict_eval.npy", np.array(loss_predict_eval_epoches))


    def test(self):
        tf.global_variables_initializer().run()
        # Loading Saved Models
        variables = tf.contrib.framework.get_variables_to_restore()
        variables_to_resotre = [v for v in variables if v.name.split('/')[0] == self.model_name + "_student"]
        # saver to save model
        self.saver = tf.train.Saver(var_list=variables_to_resotre, max_to_keep=3)
        self.load(self.model_name_save)
        # create generate for test
        g = self.dataloader.create_generator(mode='test')
        loss_test, loss_all_test = [], []
        for step in range(self.test_batch_num_per_epoch):
            batch_flow_P, batch_flow_Q, _, _, batch_weather = next(g)
            feed_dict = {self.input_history_flow_ph: batch_flow_P,
                         self.ouput_feature_flow_ph: batch_flow_Q,
                         self.input_weather_ph: batch_weather,
                         self.is_training_teacher_ph: False,
                         self.is_training_student_ph: False,
                         self.keep_prob: 1.0}
            loss, loss_all = self.sess.run([self.loss_predict, self.loss], feed_dict=feed_dict)
            loss_test.append(loss)
            loss_all_test.append(loss_all)
        print("Loss On Test Data is : %.6f,%.6f" % (np.mean(loss_test), np.mean(loss_all_test)))

    def save(self, model_name_save):
        #
        save_path = os.path.join(self.checkpoint_dir, model_name_save)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        ckpt_path = os.path.join(save_path, model_name_save)
        self.saver.save(self.sess, ckpt_path)
        print(" [*] Saving checkpoints as: {}".format(ckpt_path))
        return

    def load(self, model_name_save):
        ckpt_path = os.path.join(self.checkpoint_dir, model_name_save, model_name_save)
        print(" [*] Reading checkpoint from: {}".format(ckpt_path))
        try:
            self.saver.restore(self.sess, ckpt_path)
        except Exception as e:
            print(" [*] Loadding checkpoint From {0}, Error: {1}".format(ckpt_path, str(e)))
        return

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)

if __name__ == "__main__":
    setup_seed(0)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        my_task = TrainTask(sess, use_weather=True)
        # my_task = TrainTask(sess, use_weather=False)
        my_task.train()
        # my_task.test()


