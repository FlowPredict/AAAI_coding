# -*- coding: utf-8 -*-
import math
import tensorflow as tf
import tensorflow.contrib.slim as slim


def bn(x, is_training, scope):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=is_training,
                                        scope=scope)


def conv2d(input_, output_dim, k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02, scope_name="conv2d", padding="VALID", name='weather_weight_decay'):
    with tf.variable_scope(scope_name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        tf.add_to_collection(name=name, value=tf.nn.l2_loss(w))
        conv = tf.nn.bias_add(conv, biases)
        return conv


def fc_layer(input_, output_dim, scope_name="fc"):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(scope_name):
        matrix = tf.get_variable("weight", [shape[1], output_dim], tf.float32, tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable("bias", [output_dim], initializer=tf.constant_initializer(0.0))
        x = tf.matmul(input_, matrix) + bias
        return x

def dense(input_tensor, output_dim, activation=None, name="dense", stddev=0.02):
    x = tf.layers.dense(
        input_tensor,
        output_dim,
        activation=activation,
        name=name,
        kernel_initializer=tf.truncated_normal_initializer(stddev=stddev)
    )
    return x


def res_block(input_, is_training, scope_name='res_block'):
    output_dim = input_.get_shape()[-1]  # output_dim is same as input_dim
    with tf.variable_scope(scope_name):
        x = tf.nn.relu(bn(conv2d(input_, output_dim, 3, 3, 1, 1, scope_name='conv3x3', padding='SAME'), is_training=is_training, scope='bn1'))
        x = bn(conv2d(x, output_dim, 1, 1, 1, 1, scope_name='conv1x1', padding='SAME'), is_training=is_training, scope='bn2')
    return tf.nn.relu(input_ + x)

def layer_norm(input_, name=None):
    # Run layer normalization on the last dimension of the tensor.
    return tf.contrib.layers.layer_norm(inputs=input_, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def attention_layer(from_tensor, to_tensor, num_heads, size_per_head, scope_name="attention", initializer_range=0.02, act=None, dropout_ratio=0.0):
    # from_tensor : B x S_1 x D_1
    # to_tensor : B x S_2 x D_2
    # if from_tensor and to_tensor are the same, this is self-attention.
    with tf.variable_scope(scope_name):
        def transpose_for_scores(input_tensor, num_heads, seq_len, width):
            output_tensor = tf.reshape(input_tensor, [-1, seq_len, num_heads, width])
            output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
            return output_tensor

        from_shape = from_tensor.get_shape().as_list()
        from_seq_len, from_width = from_shape[1], from_shape[2]
        to_shape = to_tensor.get_shape().as_list()
        to_seq_len, to_width = to_shape[1], to_shape[2]

        from_tensor_2d = tf.reshape(from_tensor, [-1, from_width])  # B*S_1 x D_1
        to_tensor_2d = tf.reshape(to_tensor, [-1, to_width])  # B*S_2 x D_2

        query_layer = tf.layers.dense(
            from_tensor_2d,
            num_heads * size_per_head,
            activation=act,
            name="query",
            kernel_initializer=tf.truncated_normal_initializer(stddev=initializer_range)
        )  # B*S_1 x N*D
        key_layer = tf.layers.dense(
            to_tensor_2d,
            num_heads * size_per_head,
            activation=act,
            name="key",
            kernel_initializer=tf.truncated_normal_initializer(stddev=initializer_range)
        )  # B*S_2 x N*D
        value_layer = tf.layers.dense(
            to_tensor_2d,
            num_heads * size_per_head,
            activation=act,
            name="value",
            kernel_initializer=tf.truncated_normal_initializer(stddev=initializer_range)
        )  # B*S_2 x N*D

        # B*S_1 x N*D --> B x N x S_1 x D
        query_layer = transpose_for_scores(query_layer, num_heads, from_seq_len, size_per_head)
        # B*S_2 x N*D --> B x N x S_2 x D
        key_layer = transpose_for_scores(key_layer, num_heads, to_seq_len, size_per_head)

        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)  # B x N x S_1 x S_2
        attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(size_per_head)))  # B x N x S_1 x S_2
        attention_probs = tf.nn.softmax(attention_scores)  # B x N x S_1 x S_2
        if dropout_ratio != 0.0 and dropout_ratio is not None:
            attention_probs = tf.nn.dropout(attention_probs, 1 - dropout_ratio)

        value_layer = tf.reshape(value_layer, [-1, to_seq_len, num_heads, size_per_head])  # B x S_2 x N x D
        value_layer = tf.transpose(value_layer, [0, 2, 1, 3])  # B x N x S_2 x D
        context_layer = tf.matmul(attention_probs, value_layer)  # B x N x S_1 x D
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])  # B x S_1 x N x D
        context_layer = tf.reshape(context_layer, [-1, from_seq_len, num_heads * size_per_head])  # B x S_1 x N*D
    return context_layer


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)