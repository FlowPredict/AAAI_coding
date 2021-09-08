# -*- coding:utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import time
import numpy as np
import tensorflow as tf
from model_collection.config import cfg
from model_collection.stgcn_model import STGCNModel
from model_collection.dataloader import DataLoader
import random

class TrainTask(object):
    """ Training any model here """

    def __init__(self, sess):
        self.sess = sess
        self.model_name = cfg.train.model_name
        self.model_name_save = self.model_name + '_teacher'
        self.train_epochs = cfg.train.epochs
        self.lr_init = cfg.train.init_lr
        self.build_model()
        self.log_dir = os.path.join(cfg.train.res_save_path, "logs")
        self.checkpoint_dir = os.path.join(cfg.train.res_save_path, "checkpoint")
        self.result_loss_dir = os.path.join(cfg.train.res_save_path, "result_loss", self.model_name_save)
        if not os.path.exists(self.result_loss_dir):
            os.makedirs(self.result_loss_dir)
        self.dataloader = DataLoader()
        self.train_batch_num_per_epoch = self.dataloader.train_batch_num_per_epoch
        self.eval_batch_num_per_epoch = self.dataloader.eval_batch_num_per_epoch
        self.test_batch_num_per_epoch = self.dataloader.test_batch_num_per_epoch

    def build_model(self):
        if self.model_name == "STGCNModel":
            self.model = STGCNModel()
            self.input_history_plan_ph, self.ouput_feature_plan_ph, _, _, _, self.keep_prob, self.is_training_teacher_ph, _ = self.model.create_placeholders()
            self.model.load_weights_graph()
            _, output_plan = self.model.create_stgcn_teacher(self.input_history_plan_ph, self.keep_prob)
            self.loss = tf.reduce_mean(tf.square(self.ouput_feature_plan_ph - output_plan))
        else:
            raise("!!!!!!ModelName dosen't Exist!!!!!")

        self.global_step = tf.Variable(0, trainable=False)
        self.add_global = self.global_step.assign_add(1)
        self.lr = tf.train.exponential_decay(self.lr_init, global_step=self.global_step, decay_steps=100, decay_rate=0.8)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optim = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        """ Summary """
        loss_sum = tf.summary.scalar("loss", self.loss)
        # final summary operations
        self.summary = tf.summary.merge([loss_sum])

    def train(self):
        self.saver = tf.train.Saver(max_to_keep=3)
        # summary writer
        self.writer = tf.summary.FileWriter(os.path.join(self.log_dir, self.model_name_save + "_train"), self.sess.graph)
        self.writer_eval = tf.summary.FileWriter(os.path.join(self.log_dir, self.model_name_save + "_eval"), self.sess.graph)
        tf.global_variables_initializer().run()
        start_time = time.time()
        loss_train_epoches = []
        loss_eval_epoches = []
        for epoch in range(self.train_epochs):
            lr = sess.run(self.lr)
            print("Epoch: {:2d}, learning_rate: {:.6f}".format(epoch, lr))
            g = self.dataloader.create_generator(mode='train')
            loss_train = []
            for step in range(self.train_batch_num_per_epoch):
                train_step = sess.run(self.global_step)
                sess.run(self.add_global)
                _, _, batch_plan_P, batch_plan_Q, _ = next(g)
                feed_dict = {self.input_history_plan_ph: batch_plan_P,
                             self.ouput_feature_plan_ph: batch_plan_Q,
                             self.is_training_teacher_ph: True,
                             self.keep_prob: 1.0}
                _, summary_str, loss = self.sess.run([self.optim, self.summary, self.loss], feed_dict=feed_dict)
                self.writer.add_summary(summary_str, train_step)
                loss_train.append(loss)
                print("Epoch: [{:2d}-{:4d}], Step: {:2d}, time: {:4.2f}, loss: {:.6f}".format(epoch, step, train_step, time.time() - start_time, loss))
            loss_train_epoches.append(loss_train)

            # eval
            g = self.dataloader.create_generator(mode='eval')
            loss_eval = []
            for step in range(self.eval_batch_num_per_epoch):
                _, _, batch_plan_P, batch_plan_Q, _ = next(g)
                feed_dict = {self.input_history_plan_ph: batch_plan_P,
                             self.ouput_feature_plan_ph: batch_plan_Q,
                             self.is_training_teacher_ph: False,
                             self.keep_prob: 1.0}
                summary_str_eval, loss = self.sess.run([self.summary, self.loss], feed_dict=feed_dict)
                loss_eval.append(loss)
                self.writer_eval.add_summary(summary_str_eval, step + epoch * self.eval_batch_num_per_epoch)
            loss_eval_epoches.append(loss_eval)
            print("Epoch: {:2d}, loss on Eval Data: {:.6f}".format(epoch, np.mean(loss_eval)))
            self.save(self.model_name_save)

        np.save(self.result_loss_dir + "/loss_eval.npy", np.array(loss_eval_epoches))
        np.save(self.result_loss_dir + "/loss_train.npy", np.array(loss_train_epoches))

    def test(self):
        self.saver = tf.train.Saver(max_to_keep=3)
        tf.global_variables_initializer().run()
        # Loading Saved Models
        self.load(self.model_name_save)
        # create generate for test
        g = self.dataloader.create_generator(mode='test')
        loss_test = []
        for step in range(self.test_batch_num_per_epoch):
            _, _, batch_plan_P, batch_plan_Q, _ = next(g)
            feed_dict = {self.input_history_plan_ph: batch_plan_P,
                         self.ouput_feature_plan_ph: batch_plan_Q,
                         self.is_training_teacher_ph: False,
                         self.keep_prob: 1.0}
            loss = self.sess.run(self.loss, feed_dict=feed_dict)
            loss_test.append(loss)
        print("Loss On Test Data is : %.6f" % (np.mean(loss_test)))

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
            print(" [*] Loxitadding checkpoint From {0}, Error: {1}".format(ckpt_path, str(e)))
        return

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)

if __name__ == "__main__":
    setup_seed(5)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        my_task = TrainTask(sess)
        my_task.train()
        # my_task.test()

