# -*- coding: utf-8 -*-
import tensorflow as tf
from config import D
from model.log_configure import logger

weight_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
weight_regularizer = tf.contrib.layers.l2_regularizer(0.0001)


#  Unlike batch normalization, the instance normalization layer is applied at test time as well
#   (due to non-dependency of mini-batch)
def instance_norm(x, scope='instance_norm'):
    return tf.contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)

def relu(x):
    return tf.nn.relu(x)



# padding='SAME' ======> pad = floor[ (kernel - stride) / 2 ]  "NDHWC"
def conv2d(x, num_features, kernel=4, stride=2, pad=0, pad_type="zero",
           weight_init=tf.initializers.truncated_normal(mean=0.0, stddev=1e-3), weight_regularizer=None,
           use_bias=True, use_spectral_norm=False, dilation_rate=1, scope="conv2d", reuse=None):
    with tf.variable_scope(scope):
        if pad > 0:
            h, w = x.get_shape().as_list()[1:3]
            if h % stride == 0:
                h_pad = pad * 2
            else:
                h_pad = max(kernel - (h % stride), 0)

            if w % stride == 0:
                w_pad = pad * 2
            else:
                w_pad = max(kernel - (w % stride), 0)

            pad_top = h_pad // 2
            pad_bottom = h_pad - pad_top
            pad_left = w_pad // 2
            pad_right = w_pad - pad_left

            if pad_type == "zero":
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            if pad_type == "reflect":
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode="REFLECT")

        if use_spectral_norm:
            if scope.__contains__("generator"):
                w = tf.get_variable("kernel",
                                    shape=[kernel, kernel, x.get_shape()[-1], num_features],
                                    initializer=weight_init,
                                    regularizer=weight_regularizer)
            else:
                w = tf.get_variable("kernel",
                                    shape=[kernel, kernel, x.get_shape()[-1], num_features],
                                    initializer=weight_init,
                                    regularizer=None)

            x = tf.nn.conv2d(input=x, filter=_spectral_norm(w), dilations=[1, dilation_rate, dilation_rate, 1],
                             strides=[1, stride, stride, 1], padding="VALID")
            if use_bias:
                bias = tf.get_variable("bias", [num_features], initializer=tf.constant_initializer(.0))
                x = tf.nn.bias_add(x, bias)
        else:
            if scope.__contains__("generator"):
                x = tf.layers.conv2d(inputs=x, filters=num_features,
                                     kernel_size=kernel, kernel_initializer=weight_init,
                                     kernel_regularizer=weight_regularizer, dilation_rate=dilation_rate,
                                     strides=stride, use_bias=use_bias, reuse=reuse)
            else:
                x = tf.layers.conv2d(inputs=x, filters=num_features,
                                     kernel_size=kernel, kernel_initializer=weight_init,
                                     kernel_regularizer=None, dilation_rate=dilation_rate,
                                     strides=stride, use_bias=use_bias, reuse=reuse)

        return x



# padding='SAME' ======> pad = floor[ (kernel - stride) / 2 ]
def conv3d(x, num_features, kernel=4, stride=2, pad=0, pad_type="zero",
           weight_init=tf.initializers.random_normal(mean=0.2, stddev=1e-1), weight_regularizer=None,
           use_bias=True, use_spectral_norm=False, dilation_rate=1, scope="conv3d", reuse=None):
    with tf.variable_scope(scope):
        if pad > 0:
            d, h, w = x.get_shape().as_list()[1:4]
            if d % stride == 0:
                d_pad = pad * 2
            else:
                d_pad = max(kernel - (d % stride), 0)
            if h % stride == 0:
                h_pad = pad * 2
            else:
                h_pad = max(kernel - (h % stride), 0)
            if w % stride == 0:
                w_pad = pad * 2
            else:
                w_pad = max(kernel - (w % stride), 0)

            pad_ahead = d_pad // 2
            pad_back = d_pad - pad_ahead
            pad_top = h_pad // 2
            pad_bottom = h_pad - pad_top
            pad_left = w_pad // 2
            pad_right = w_pad - pad_left

            if pad_type == "zero":
                x = tf.pad(x, [[0, 0], [pad_ahead, pad_back], [pad_top, pad_bottom],
                               [pad_left, pad_right], [0, 0]])
            if pad_type == "reflect":
                x = tf.pad(x, [[0, 0], [pad_ahead, pad_back], [pad_top, pad_bottom],
                               [pad_left, pad_right], [0, 0]], mode="REFLECT")

        if use_spectral_norm:
            if scope.__contains__("generator"):
                w = tf.get_variable("kernel",
                                    shape=[kernel, kernel, kernel, x.get_shape()[-1], num_features],
                                    initializer=weight_init,
                                    regularizer=weight_regularizer)
            else:
                w = tf.get_variable("kernel",
                                    shape=[kernel, kernel, kernel, x.get_shape()[-1], num_features],
                                    initializer=weight_init,
                                    regularizer=None)

            x = tf.nn.conv3d(input, filter=_spectral_norm(w), strides=[1, stride, stride, stride, 1], padding="VALID",
                             dilations=[1, dilation_rate, dilation_rate, dilation_rate, 1])

            if use_bias:
                bias = tf.get_variable("bias", [num_features], initializer=tf.constant_initializer(.0))
                x = tf.nn.bias_add(x, bias)
        else:
            if scope.__contains__("generator"):
                x = tf.layers.conv3d(inputs=x, filters=num_features,
                                     kernel_size=kernel, kernel_initializer=weight_init,
                                     kernel_regularizer=weight_regularizer, dilation_rate=dilation_rate,
                                     strides=stride, use_bias=use_bias, reuse=reuse)
            else:
                x = tf.layers.conv3d(inputs=x, filters=num_features,
                                     kernel_size=kernel, kernel_initializer=weight_init,
                                     kernel_regularizer=None, dilation_rate=dilation_rate,
                                     strides=stride, use_bias=use_bias, reuse=reuse)

        return x



def _shape_list(x):
    """Return list of dims, statically where possible."""
    static = x.get_shape().as_list()
    shape = tf.shape(x)
    ret = []
    for i, static_dim in enumerate(static):
        dim = static_dim or shape[i]
        ret.append(dim)
    return ret


def no_local_block(x_init, cut_dim, scope="no_local_block", reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        _, height, width, dim = _shape_list(x_init)

        proj_query = conv2d(x_init, cut_dim, kernel=1, stride=1, pad=0, scope="conv2d_query")  # (b, h, w, c)
        proj_query = tf.reshape(proj_query, (-1, width * height, cut_dim))  # (b, hxwxc, 1)

        proj_key = conv2d(x_init, cut_dim, kernel=1, stride=1, pad=0, scope="conv2d_key")  # (b, h, w, c)
        proj_key = tf.transpose(tf.reshape(proj_key, (-1, width * height, cut_dim)), (0, 2, 1))  # (b, 1, hxwxc)

        energy = tf.matmul(proj_query, proj_key)  # transpose check
        attention = tf.nn.softmax(energy)  # (b, hxwxc, hxwxc)  axis=-1 column sum = 1

        proj_value = conv2d(x_init, cut_dim, kernel=1, stride=1, pad=0, scope="conv2d_value")  # (b, h, w, c)
        proj_value = tf.reshape(proj_value, (-1, width * height, cut_dim))  # (b, hxwxc, 1)

        out = tf.matmul(attention, proj_value)  # (b, hxwxc, 1)
        out = tf.reshape(out, (-1, height, width, cut_dim))  # (b, h, w, c)

        out = conv2d(out, dim, kernel=1, stride=1, pad=0, scope="conv2d_final")  # (b, h, w, c)
        gamma = tf.get_variable('gamma', shape=(), dtype=tf.float32, initializer=tf.zeros_initializer())
        out = gamma * out + x_init

        return out


def no_local_block_3d(x_init, cut_dim, scope="no_local_block_3d", reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        _, time, height, width, dim = _shape_list(x_init)

        proj_query = conv3d(x_init, cut_dim, kernel=1, stride=1, pad=0, scope="conv3d_query")  # (b, t, h, w, c)
        proj_query = tf.reshape(proj_query, (-1, time * width * height, cut_dim))  # (b, txhxw, c)

        proj_key = conv3d(x_init, cut_dim, kernel=1, stride=1, pad=0, scope="conv3d_key")  # (b, t, h, w, c)
        proj_key = tf.transpose(tf.reshape(proj_key, (-1, time * width * height, cut_dim)), (0, 2, 1))  # (b, c, txhxw)

        energy = tf.matmul(proj_query, proj_key)  # transpose check
        attention = tf.nn.softmax(energy)  # (b, hxwxc, hxwxc)  axis=-1 column sum = 1

        proj_value = conv3d(x_init, cut_dim, kernel=1, stride=1, pad=0, scope="conv3d_value")  # (b, t, h, w, c)
        proj_value = tf.reshape(proj_value, (-1, time * width * height, cut_dim))  # (b, txhxw, c)

        out = tf.matmul(attention, proj_value)  # (b, txhxw, c)
        out = tf.reshape(out, (-1, time, height, width, cut_dim))  # (b, t, h, w, c)

        out = conv3d(out, dim, kernel=1, stride=1, pad=0, scope="conv3d_final")  # (b, h, w, c)
        gamma = tf.get_variable('gamma', shape=(), dtype=tf.float32, initializer=tf.zeros_initializer())
        out = gamma * out + x_init

        return out


def dense_block(x_init, n_features, n_blocks=4,
                ops_type='conv2d', pad=1, cut_dim=8, use_bias=True, use_spectral_norm=False, scope="dense_block"):
    with tf.variable_scope(scope):
        x_concat = [x_init]
        for i in range(n_blocks):
            x = tf.concat(x_concat, axis=-1)

            if ops_type == 'conv2d':
                x = conv2d(x, n_features // 2, kernel=3, stride=1, pad=pad,
                           use_bias=use_bias, use_spectral_norm=use_spectral_norm, scope="conv2d-{}".format(i))
            elif ops_type == 'nolocal2d':
                x = no_local_block(x, cut_dim, scope="nolocal-{}".format(i))
            elif ops_type == 'conv3d':
                x = conv2d(x, n_features // 2, kernel=3, stride=1, pad=pad,
                           use_bias=use_bias, use_spectral_norm=use_spectral_norm, scope="conv3d-{}".format(i))
            elif ops_type == 'nolocal3d':
                x = no_local_block_3d(x, cut_dim, scope="nolocal_3d-{}".format(i))

            x = instance_norm(x, scope="instance_norm-{}".format(i))
            x = relu(x)
            x_concat.append(x)

        x = tf.concat(x_concat, axis=-1)

        if ops_type == 'conv2d' or ops_type == 'nolocal2d':
            x = conv2d(x, n_features, kernel=3, stride=1, pad=pad,
                       use_bias=use_bias, use_spectral_norm=use_spectral_norm, scope="conv2d_proj")
        elif ops_type == 'conv3d' or ops_type == 'nolocal3d':
            x = conv3d(x, n_features, kernel=3, stride=1, pad=pad,
                       use_bias=use_bias, use_spectral_norm=use_spectral_norm, scope="conv3d_proj")

        x = instance_norm(x, scope="instance_norm_proj")
        return x


def residual_dense_block(x_init, n_features, n_blocks=3, n_in_layers=4, beta=0.2,
                         ops_type='conv2d', pad=1, cut_dim=8, use_bias=True, use_spectral_norm=False,
                         scope="residual_dense_block"):
    with tf.variable_scope(scope):
        x = x_init
        for i in range(n_blocks):
            tempx = dense_block(x, n_features, n_blocks=n_in_layers, ops_type=ops_type, pad=pad, cut_dim=cut_dim,
                                use_bias=use_bias, use_spectral_norm=use_spectral_norm,
                                scope="dense_block-{}".format(i))
            x = x + tempx * beta
        x = x_init + x * beta
        return x
