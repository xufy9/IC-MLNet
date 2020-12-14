# -*- coding: utf-8 -*-
import re
import math
from model.ops import *


def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % "tower", '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


class MultiStageModel:
    def __init__(self, params):
        """
        Args:
          train_variables:
          loss_list: dict (3)  {loss, final_loss, smooth_loss}
          acc_list:  dict (18)
                       {'mae', 'bias', 'rmse', 'corr', 'ssmean', 'ssmin', 'psnr','ssim',
                        'acc', 'hss', 'hk', 'ssaccmax',
                        'sunts', 'smallts', 'midts', 'largets', 'stormts', 'hugestormts'}
        """
        self.params = params
        # self.crossmem1 = CrossMember('crossmem1')
        # self.crossmem2 = CrossMember('crossmem2')
        # self.crossmem3 = CrossMember('crossmem3')
        # self.h_gamma = tf.Variable(0.0, trainable=True, name='plush_gamma')
        # self.m_gamma = tf.Variable(0.0, trainable=True, name='plusm_gamma')
        # self.r_gamma = tf.Variable(1.0, trainable=True, name='res_gamma')
        self.train_variables = None
        self.loss_dic = {}
        self.acc_dic = {}

    def singlelayer_redense(self, xs, training=True, reuse=False):
        """
            param:
            xs:  tensor (B, h, w, sC)   sorted, 0th the best
            return
            smooth_o:   tensor (B, h, w, outC)
            fine_o:   tensor (B, h, w, outC)
        """
        num_filters = self.params.get('num_filters', 1)
        cut_dim = self.params.get('cut_dim', 8)
        dense_block = self.params.get('dense_block', 3)
        in_dense_layers = self.params.get('in_dense_layers', 4)
        res_dense_block = self.params.get('res_dense_block', 4)
        ops_type = self.params.get('block_type', 'nolocal')
        merge_type = self.params.get('merge_type', 'add')

        with tf.variable_scope('multilayer', reuse=reuse):
            x = tf.nn.top_k(xs, k=xs.get_shape().as_list()[-1]).values
            logger.debug("x %s shape: %s", x.name, x.get_shape().as_list())
            # tf.summary.image("1-sort_sample", self._concact_features(x))
            mid = x.get_shape()[-1] // 2 + 1
            median_x = tf.expand_dims(x[:, :, :, mid], axis=-1)  # (b, h, w, 1)

            l_x = tf.where(x >= 0.1, x, tf.zeros_like(x))
            layers_input_or_feature = [l_x]  # (B, h, w, sC)

            l_x = conv2d(l_x, num_filters, 3, 1, pad=1, scope="conv2ds0")
            layers_input_or_feature.append(l_x)  # [(B, h, w, sC), (B, h, w, c)]

            for i in range(res_dense_block):
                l_x = residual_dense_block(l_x, num_filters, n_blocks=dense_block, n_in_layers=in_dense_layers,
                                           beta=0.2,
                                           ops_type=ops_type, pad=1, cut_dim=cut_dim, use_bias=True,
                                           scope="residual_dense_block-st{}".format(i))

            l_x = conv2d(l_x, 1, 1, 1, scope="conv2d_f")

            if merge_type == 'add':
                res_gamma = tf.get_variable('res_gamma', shape=(), dtype=tf.float32,
                                            initializer=tf.zeros_initializer())
                tf.summary.scalar("res_gamma", res_gamma)
                outx = l_x * res_gamma + median_x
            elif merge_type == 'concat':
                outx = conv2d(tf.concat([l_x, median_x], axis=-1), 1, 1, 1, scope="conv2d_out")

            tf.summary.image("outlx", outx)

            self.train_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        return outx, layers_input_or_feature, [l_x], [outx]

    def twolayer_redense(self, xs, training=True, reuse=False):
        """
            param:
            xs:  tensor (B, h, w, sC)   sorted, 0th the best
            return
            smooth_o:   tensor (B, h, w, outC)
            fine_o:   tensor (B, h, w, outC)
        """
        num_filters = self.params.get('num_filters', 1)
        cut_dim = self.params.get('cut_dim', 8)
        dense_block = self.params.get('dense_block', 3)
        in_dense_layers = self.params.get('in_dense_layers', 4)
        res_dense_block = self.params.get('res_dense_block', 4)
        ops_type = self.params.get('block_type', 'nolocal')
        merge_type = self.params.get('merge_type', 'add')

        with tf.variable_scope('multilayer', reuse=reuse):
            x = tf.nn.top_k(xs, k=xs.get_shape().as_list()[-1]).values
            logger.debug("x %s shape: %s", x.name, x.get_shape().as_list())
            # tf.summary.image("1-sort_sample", self._concact_features(x))
            mid = x.get_shape()[-1] // 2 + 1
            median_x = tf.expand_dims(x[:, :, :, mid], axis=-1)  # (b, h, w, 1)

            m_mid = tf.where(median_x >= 10.1, median_x, tf.zeros_like(median_x))
            s_mid = tf.where(median_x >= 0.1, median_x, tf.zeros_like(median_x))

            m_x = tf.where(x >= 10.1, x, tf.zeros_like(x))
            l_x = tf.where(x >= 0.1, x, tf.zeros_like(x))

            layers_input_or_feature = []
            layers_input_or_feature.append(tf.concat([m_x, l_x], axis=-1))  # (B, h, w, sC*2)

            m_x = conv2d(m_x, num_filters, 3, 1, pad=1, scope="conv2dm0")
            l_x = conv2d(l_x, num_filters, 3, 1, pad=1, scope="conv2ds0")

            layers_input_or_feature.append(tf.concat([m_x, l_x], axis=-1))  # [(B, h, w, sC*2), (B, h, w, c*2)]

            for i in range(res_dense_block):
                m_x = residual_dense_block(m_x, num_filters, n_blocks=dense_block, n_in_layers=in_dense_layers,
                                           beta=0.2,
                                           ops_type=ops_type, pad=1, cut_dim=cut_dim, use_bias=True,
                                           scope="residual_dense_block-m{}".format(i))
            for i in range(res_dense_block):
                l_x = residual_dense_block(l_x, num_filters, n_blocks=dense_block, n_in_layers=in_dense_layers,
                                           beta=0.2,
                                           ops_type=ops_type, pad=1, cut_dim=cut_dim, use_bias=True,
                                           scope="residual_dense_block-s{}".format(i))

            if merge_type == 'add':
                logger.debug("merge_type=%s", merge_type)
                layers_gamma = tf.get_variable('layers_gamma', shape=(), dtype=tf.float32,
                                               initializer=tf.zeros_initializer())
                tf.summary.scalar("layers_gamma", layers_gamma)
                l_x = m_x * layers_gamma + l_x
            elif merge_type == 'concat':
                logger.debug("merge_type=%s", merge_type)
                l_x = tf.concat([m_x, l_x], axis=-1)

            m_x = conv2d(m_x, 1, 1, 1, scope="conv2d_mf")
            l_x = conv2d(l_x, 1, 1, 1, scope="conv2d_sf")

            if merge_type == 'add':
                res_gamma = tf.get_variable('res_gamma', shape=(), dtype=tf.float32,
                                            initializer=tf.zeros_initializer())
                tf.summary.scalar("res_gamma", res_gamma)
                outmx = m_x * res_gamma + m_mid
                outsx = l_x * res_gamma + s_mid
            elif merge_type == 'concat':
                outmx = conv2d(tf.concat([m_x, m_mid], axis=-1), 1, 1, 1, scope="conv2d_outf")
                outsx = conv2d(tf.concat([l_x, s_mid], axis=-1), 1, 1, 1, scope="conv2d_outs")

            tf.summary.image("outsx", outsx)

            self.train_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        return outsx, layers_input_or_feature, [m_x, l_x], [outsx, outmx]

    def threelayer_redense(self, xs, training=True, reuse=False):
        """
            param:
            xs:  tensor (B, h, w, sC)   sorted, 0th the best
            return
            smooth_o:   tensor (B, h, w, outC)
            fine_o:   tensor (B, h, w, outC)
        """
        num_filters = self.params.get('num_filters', 1)
        cut_dim = self.params.get('cut_dim', 8)
        dense_block = self.params.get('dense_block', 3)
        in_dense_layers = self.params.get('in_dense_layers', 4)
        res_dense_block = self.params.get('res_dense_block', 4)
        ops_type = self.params.get('block_type', 'nolocal')
        merge_type = self.params.get('merge_type', 'add')

        with tf.variable_scope('multilayer', reuse=reuse):
            x = tf.nn.top_k(xs, k=xs.get_shape().as_list()[-1]).values
            logger.debug("x %s shape: %s", x.name, x.get_shape().as_list())
            # tf.summary.image("1-sort_sample", self._concact_features(x))
            mid = x.get_shape()[-1] // 2 + 1
            median_x = tf.expand_dims(x[:, :, :, mid], axis=-1)  # (b, h, w, 1)

            # extract feature
            h_mid = tf.where(median_x >= 25.1, median_x, tf.zeros_like(median_x))
            m_mid = tf.where(median_x >= 10.1, median_x, tf.zeros_like(median_x))
            s_mid = tf.where(median_x >= 0.1, median_x, tf.zeros_like(median_x))

            h_x = tf.where(x >= 25.1, x, tf.zeros_like(x))
            m_x = tf.where(x >= 10.1, x, tf.zeros_like(x))
            l_x = tf.where(x >= 0.1, x, tf.zeros_like(x))

            layers_input_or_feature = []
            layers_input_or_feature.append(tf.concat([h_x, m_x, l_x], axis=-1))  # (B, h, w, sC*4)

            h_x = conv2d(h_x, num_filters, 3, 1, pad=1, scope="conv2dh0")
            m_x = conv2d(m_x, num_filters, 3, 1, pad=1, scope="conv2dm0")
            l_x = conv2d(l_x, num_filters, 3, 1, pad=1, scope="conv2ds0")

            layers_input_or_feature.append(tf.concat([h_x, m_x, l_x], axis=-1))  # [(B, h, w, sC*4), (B, h, w, c*4)]

            # bias correction
            for i in range(res_dense_block):
                h_x = residual_dense_block(h_x, num_filters, n_blocks=dense_block, n_in_layers=in_dense_layers,
                                           beta=0.2,
                                           ops_type=ops_type, pad=1, cut_dim=cut_dim, use_bias=True,
                                           scope="residual_dense_block-h{}".format(i))
            for i in range(res_dense_block):
                m_x = residual_dense_block(m_x, num_filters, n_blocks=dense_block, n_in_layers=in_dense_layers,
                                           beta=0.2,
                                           ops_type=ops_type, pad=1, cut_dim=cut_dim, use_bias=True,
                                           scope="residual_dense_block-m{}".format(i))
            for i in range(res_dense_block):
                l_x = residual_dense_block(l_x, num_filters, n_blocks=dense_block, n_in_layers=in_dense_layers,
                                           beta=0.2,
                                           ops_type=ops_type, pad=1, cut_dim=cut_dim, use_bias=True,
                                           scope="residual_dense_block-s{}".format(i))

            # merge feature
            if merge_type == 'add':
                logger.debug("merge_type=%s", merge_type)
                layers_gamma = tf.get_variable('layers_gamma', shape=(), dtype=tf.float32,
                                               initializer=tf.zeros_initializer())
                tf.summary.scalar("layers_gamma", layers_gamma)
                m_x = h_x * layers_gamma + m_x
                l_x = m_x * layers_gamma + l_x
            elif merge_type == 'concat':
                logger.debug("merge_type=%s", merge_type)
                m_x = tf.concat([h_x, m_x], axis=-1)
                l_x = tf.concat([m_x, l_x], axis=-1)

            h_x = conv2d(h_x, 1, 1, 1, scope="conv2d_hf")
            m_x = conv2d(m_x, 1, 1, 1, scope="conv2d_mf")
            l_x = conv2d(l_x, 1, 1, 1, scope="conv2d_sf")

            if merge_type == 'add':
                res_gamma = tf.get_variable('res_gamma', shape=(), dtype=tf.float32,
                                            initializer=tf.zeros_initializer())
                tf.summary.scalar("res_gamma", res_gamma)
                outhx = h_x * res_gamma + h_mid
                outmx = m_x * res_gamma + m_mid
                outlx = l_x * res_gamma + s_mid
            elif merge_type == 'concat':
                outhx = conv2d(tf.concat([h_x, h_mid], axis=-1), 1, 1, 1, scope="conv2d_outh")
                outmx = conv2d(tf.concat([m_x, m_mid], axis=-1), 1, 1, 1, scope="conv2d_outf")
                outlx = conv2d(tf.concat([l_x, s_mid], axis=-1), 1, 1, 1, scope="conv2d_outs")

            tf.summary.image("outlx", outlx)

            self.train_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        return outlx, layers_input_or_feature, [l_x, m_x, h_x], [outlx, outmx, outhx]

    def fivelayer_redense(self, xs, training=True, reuse=False):
        """
            param:
            xs:  tensor (B, h, w, sC)   sorted, 0th the best
            return
            smooth_o:   tensor (B, h, w, outC)
            fine_o:   tensor (B, h, w, outC)
        """
        num_filters = self.params.get('num_filters', 1)
        cut_dim = self.params.get('cut_dim', 8)
        dense_block = self.params.get('dense_block', 3)
        in_dense_layers = self.params.get('in_dense_layers', 4)
        res_dense_block = self.params.get('res_dense_block', 4)
        ops_type = self.params.get('block_type', 'nolocal')
        merge_type = self.params.get('merge_type', 'add')

        with tf.variable_scope('multilayer', reuse=reuse):
            x = tf.nn.top_k(xs, k=xs.get_shape().as_list()[-1]).values
            logger.debug("x %s shape: %s", x.name, x.get_shape().as_list())
            # # tf.summary.image("1-sort_sample", self._concact_features(x))
            mid = xs.get_shape()[-1] // 2 + 1
            median_x = tf.expand_dims(x[:, :, :, mid], axis=-1)  # (b, h, w, 1)
            # median_x = tf.expand_dims(tf.nn.top_k(xs, k=xs.get_shape().as_list()[-1]).values[:, :, :, mid], axis=-1)

            # extract feature
            st_mid = tf.where(median_x >= 100.1, median_x, tf.zeros_like(median_x))
            v_mid = tf.where(median_x >= 50.1, median_x, tf.zeros_like(median_x))
            h_mid = tf.where(median_x >= 25.1, median_x, tf.zeros_like(median_x))
            m_mid = tf.where(median_x >= 10.1, median_x, tf.zeros_like(median_x))
            l_mid = tf.where(median_x >= 0.1, median_x, tf.zeros_like(median_x))

            st_x = tf.where(x >= 100.1, x, tf.zeros_like(x))  # (B, h, w, sC)
            v_x = tf.where(x >= 50.1, x, tf.zeros_like(x))
            h_x = tf.where(x >= 25.1, x, tf.zeros_like(x))
            m_x = tf.where(x >= 10.1, x, tf.zeros_like(x))
            l_x = tf.where(x >= 0.1, x, tf.zeros_like(x))

            layers_input_or_feature = []
            layers_input_or_feature.append(tf.concat([st_x, h_x, m_x, l_x], axis=-1))  # (B, h, w, sC*4)

            st_x = conv2d(st_x, num_filters, 3, 1, pad=1, scope="fext-st")
            v_x = conv2d(v_x, num_filters, 3, 1, pad=1, scope="fext-v")
            h_x = conv2d(h_x, num_filters, 3, 1, pad=1, scope="fext-h")
            m_x = conv2d(m_x, num_filters, 3, 1, pad=1, scope="fext-m")
            l_x = conv2d(l_x, num_filters, 3, 1, pad=1, scope="fext-l")

            layers_input_or_feature.append(
                tf.concat([st_x, h_x, m_x, l_x], axis=-1))  # [(B, h, w, sC*4), (B, h, w, c*4)]

            # bias correction
            for i in range(res_dense_block):
                st_x = residual_dense_block(st_x, num_filters, n_blocks=dense_block, n_in_layers=in_dense_layers,
                                            beta=0.2,
                                            ops_type=ops_type, pad=1, cut_dim=cut_dim, use_bias=True,
                                            scope="residual_dense_block-st{}".format(i))
            for i in range(res_dense_block):
                v_x = residual_dense_block(v_x, num_filters, n_blocks=dense_block, n_in_layers=in_dense_layers,
                                           beta=0.2,
                                           ops_type=ops_type, pad=1, cut_dim=cut_dim, use_bias=True,
                                           scope="residual_dense_block-v{}".format(i))
            for i in range(res_dense_block):
                h_x = residual_dense_block(h_x, num_filters, n_blocks=dense_block, n_in_layers=in_dense_layers,
                                           beta=0.2,
                                           ops_type=ops_type, pad=1, cut_dim=cut_dim, use_bias=True,
                                           scope="residual_dense_block-h{}".format(i))
            for i in range(res_dense_block):
                m_x = residual_dense_block(m_x, num_filters, n_blocks=dense_block, n_in_layers=in_dense_layers,
                                           beta=0.2,
                                           ops_type=ops_type, pad=1, cut_dim=cut_dim, use_bias=True,
                                           scope="residual_dense_block-m{}".format(i))
            for i in range(res_dense_block):
                l_x = residual_dense_block(l_x, num_filters, n_blocks=dense_block, n_in_layers=in_dense_layers,
                                           beta=0.2,
                                           ops_type=ops_type, pad=1, cut_dim=cut_dim, use_bias=True,
                                           scope="residual_dense_block-l{}".format(i))

            # merge feature
            if merge_type == 'add':
                logger.debug("merge_type=%s", merge_type)
                layers_gamma = tf.get_variable('layers_gamma', shape=(), dtype=tf.float32,
                                               initializer=tf.zeros_initializer())
                tf.summary.scalar("layers_gamma", layers_gamma)
                v_x = st_x * layers_gamma + v_x
                h_x = v_x * layers_gamma + h_x
                m_x = h_x * layers_gamma + m_x
                l_x = m_x * layers_gamma + l_x
            elif merge_type == 'concat':
                logger.debug("merge_type=%s", merge_type)
                v_x = tf.concat([st_x, v_x], axis=-1)
                h_x = tf.concat([v_x, h_x], axis=-1)
                m_x = tf.concat([h_x, m_x], axis=-1)
                l_x = tf.concat([m_x, l_x], axis=-1)

            st_x = conv2d(st_x, 1, 1, 1, scope="fmer_stf")
            v_x = conv2d(v_x, 1, 1, 1, scope="fmer_vf")
            h_x = conv2d(h_x, 1, 1, 1, scope="fmer_hf")
            m_x = conv2d(m_x, 1, 1, 1, scope="fmer_mf")
            l_x = conv2d(l_x, 1, 1, 1, scope="fmer_lf")

            if merge_type == 'add':
                res_gamma = tf.get_variable('res_gamma', shape=(), dtype=tf.float32,
                                            initializer=tf.zeros_initializer())
                tf.summary.scalar("res_gamma", res_gamma)
                outstx = st_x * res_gamma + st_mid
                outvx = v_x * res_gamma + v_mid
                outhx = h_x * res_gamma + h_mid
                outmx = m_x * res_gamma + m_mid
                outlx = l_x * res_gamma + l_mid
            elif merge_type == 'concat':
                outstx = conv2d(tf.concat([st_x, st_mid], axis=-1), 1, 1, 1, scope="conv2d_outst")
                outvx = conv2d(tf.concat([v_x, v_mid], axis=-1), 1, 1, 1, scope="conv2d_outv")
                outhx = conv2d(tf.concat([h_x, h_mid], axis=-1), 1, 1, 1, scope="conv2d_outh")
                outmx = conv2d(tf.concat([m_x, m_mid], axis=-1), 1, 1, 1, scope="conv2d_outf")
                outlx = conv2d(tf.concat([l_x, l_mid], axis=-1), 1, 1, 1, scope="conv2d_outs")

            # tf.summary.image("outlx", outlx)

            self.train_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        return outlx, layers_input_or_feature, [l_x, m_x, h_x, v_x, st_x], [outlx, outmx, outhx, outvx, outstx]

    def multilayer_redense(self, xs, training=True, reuse=False):
        """
            param:
            xs:  tensor (B, h, w, sC)   sorted, 0th the best
            return
            smooth_o:   tensor (B, h, w, outC)
            fine_o:   tensor (B, h, w, outC)
        """
        num_filters = self.params.get('num_filters', 1)
        cut_dim = self.params.get('cut_dim', 8)
        dense_block = self.params.get('dense_block', 3)
        in_dense_layers = self.params.get('in_dense_layers', 4)
        res_dense_block = self.params.get('res_dense_block', 4)
        ops_type = self.params.get('block_type', 'nolocal')
        merge_type = self.params.get('merge_type', 'add')

        with tf.variable_scope('multilayer', reuse=reuse):
            x = tf.nn.top_k(xs, k=xs.get_shape().as_list()[-1]).values
            logger.debug("x %s shape: %s", x.name, x.get_shape().as_list())
            # # tf.summary.image("1-sort_sample", self._concact_features(x))
            mid = xs.get_shape()[-1] // 2 + 1
            median_x = tf.expand_dims(x[:, :, :, mid], axis=-1)  # (b, h, w, 1)
            # median_x = tf.expand_dims(tf.nn.top_k(xs, k=xs.get_shape().as_list()[-1]).values[:, :, :, mid], axis=-1)

            # extract feature
            st_mid = tf.where(median_x >= 50.1, median_x, tf.zeros_like(median_x))
            h_mid = tf.where(median_x >= 25.1, median_x, tf.zeros_like(median_x))
            m_mid = tf.where(median_x >= 10.1, median_x, tf.zeros_like(median_x))
            s_mid = tf.where(median_x >= 0.1, median_x, tf.zeros_like(median_x))

            st_x = tf.where(x >= 50.1, x, tf.zeros_like(x))  # (B, h, w, sC)
            h_x = tf.where(x >= 25.1, x, tf.zeros_like(x))
            m_x = tf.where(x >= 10.1, x, tf.zeros_like(x))
            l_x = tf.where(x >= 0.1, x, tf.zeros_like(x))

            layers_sorted_and_feature = []
            layers_sorted_and_feature.append(tf.concat([st_x, h_x, m_x, l_x], axis=-1))  # (B, h, w, sC*4)

            st_x = conv2d(st_x, num_filters, 1, 1, pad=0, scope="conv2dst0")
            h_x = conv2d(h_x, num_filters, 1, 1, pad=0, scope="conv2dh0")
            m_x = conv2d(m_x, num_filters, 1, 1, pad=0, scope="conv2dm0")
            l_x = conv2d(l_x, num_filters, 1, 1, pad=0, scope="conv2ds0")

            layers_sorted_and_feature.append(
                tf.concat([st_x, h_x, m_x, l_x], axis=-1))  # [(B, h, w, sC*4), (B, h, w, c*4)]

            # bias correction
            for i in range(res_dense_block):
                st_x = residual_dense_block(st_x, num_filters, n_blocks=dense_block, n_in_layers=in_dense_layers,
                                            beta=0.2,
                                            ops_type=ops_type, pad=1, cut_dim=cut_dim, use_bias=True,
                                            scope="residual_dense_block-st{}".format(i))
            for i in range(res_dense_block):
                h_x = residual_dense_block(h_x, num_filters, n_blocks=dense_block, n_in_layers=in_dense_layers,
                                           beta=0.2,
                                           ops_type=ops_type, pad=1, cut_dim=cut_dim, use_bias=True,
                                           scope="residual_dense_block-h{}".format(i))
            for i in range(res_dense_block):
                m_x = residual_dense_block(m_x, num_filters, n_blocks=dense_block, n_in_layers=in_dense_layers,
                                           beta=0.2,
                                           ops_type=ops_type, pad=1, cut_dim=cut_dim, use_bias=True,
                                           scope="residual_dense_block-m{}".format(i))
            for i in range(res_dense_block):
                l_x = residual_dense_block(l_x, num_filters, n_blocks=dense_block, n_in_layers=in_dense_layers,
                                           beta=0.2,
                                           ops_type=ops_type, pad=1, cut_dim=cut_dim, use_bias=True,
                                           scope="residual_dense_block-s{}".format(i))

            # merge feature
            if merge_type == 'add':
                logger.debug("merge_type=%s", merge_type)
                layers_gamma = tf.get_variable('layers_gamma', shape=(), dtype=tf.float32,
                                               initializer=tf.zeros_initializer())
                tf.summary.scalar("layers_gamma", layers_gamma)
                h_x = st_x * layers_gamma + h_x
                m_x = h_x * layers_gamma + m_x
                l_x = m_x * layers_gamma + l_x
            elif merge_type == 'concat':
                logger.debug("merge_type=%s", merge_type)
                h_x = tf.concat([st_x, h_x], axis=-1)
                m_x = tf.concat([h_x, m_x], axis=-1)
                l_x = tf.concat([m_x, l_x], axis=-1)

            st_x = conv2d(st_x, 1, 1, 1, scope="conv2d_stf")
            h_x = conv2d(h_x, 1, 1, 1, scope="conv2d_hf")
            m_x = conv2d(m_x, 1, 1, 1, scope="conv2d_mf")
            l_x = conv2d(l_x, 1, 1, 1, scope="conv2d_sf")

            if merge_type == 'add':
                res_gamma = tf.get_variable('res_gamma', shape=(), dtype=tf.float32,
                                            initializer=tf.zeros_initializer())
                tf.summary.scalar("res_gamma", res_gamma)
                outstx = st_x * res_gamma + st_mid
                outhx = h_x * res_gamma + h_mid
                outmx = m_x * res_gamma + m_mid
                outlx = l_x * res_gamma + s_mid
            elif merge_type == 'concat':
                outstx = conv2d(tf.concat([st_x, st_mid], axis=-1), 1, 1, 1, scope="conv2d_outst")
                outhx = conv2d(tf.concat([h_x, h_mid], axis=-1), 1, 1, 1, scope="conv2d_outh")
                outmx = conv2d(tf.concat([m_x, m_mid], axis=-1), 1, 1, 1, scope="conv2d_outf")
                outlx = conv2d(tf.concat([l_x, s_mid], axis=-1), 1, 1, 1, scope="conv2d_outs")

            # tf.summary.image("outlx", outlx)

            self.train_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        return outlx, layers_sorted_and_feature, [l_x, m_x, h_x, st_x], [outlx, outmx, outhx, outstx]

    def compute_loss(self, labels, predicts, surface_mask, layers_out):
        """
           !!!invalid value has been -5
            Args:
              predicts:   tensor shape:[B, H, W, outC]
              labels:   tensor shape:[B, H, W, outC]
              surface_mask:   tensor shape:[B, H, W, outC]   labels >= 0.0  ie. surface=1 sea=0
              layers_out: list [tensor]  M:[B, H, W, outC]

            :return:  loss, loss_train_arr[loss, final_loss, smooth_loss]
            Returns:
              output: list, [batch_size x out_size x out_size x N]  len: mem_block
        """
        with tf.name_scope("compute_loss"):
            labels_classed = [tf.where(labels >= 0.1, labels, tf.zeros_like(labels)),
                              tf.where(labels >= 10.1, labels, tf.zeros_like(labels)),
                              tf.where(labels >= 25.1, labels, tf.zeros_like(labels)),
                              tf.where(labels >= 50.1, labels, tf.zeros_like(labels)),
                              tf.where(labels >= 100.1, labels, tf.zeros_like(labels))]

            layers_loss = tf.convert_to_tensor(0, dtype=tf.float32)
            M = len(layers_out)  # [r, m, h, s]
            for i in range(1, M, 1):
                layers_loss += self._count_MSE(labels_classed[i], layers_out[i], surface_mask)

            #  ------ loss ----
            #  predicts_loss + layers_loss

            mse_w, mae_w = 0.7, 0.6
            layer_w, predic_w = 0.23, 0.77

            predicts_loss = self._count_MSE(labels, predicts, surface_mask)
            # * mse_w + self._count_MAE(labels, predicts, surface_mask) * mae_w

            logger.debug("predicts_loss %s shape: %s", predicts_loss.name, predicts_loss.get_shape().as_list())

            loss = layers_loss * layer_w + predicts_loss * predic_w

            loss_dic = {"loss": loss, "final_loss": predicts_loss, "layer_loss": layers_loss}

            return loss_dic

    # mae   Range: 0 to infinity.  Perfect score: 0
    def _count_MAE(self, labels, predicts, surface_mask):
        return self._reduce_mean(tf.abs(predicts - labels), mask=surface_mask, name='mae')

    # mse   Range: 0 to infinity.  Perfect score: 0
    def _count_MSE(self, labels, predicts, surface_mask):
        return self._reduce_mean(tf.squared_difference(predicts, labels), mask=surface_mask, name='mse')

    def _reduce_mean(self, arr, mask, axis=None, name=None):
        """
        :param arr:  arr has multiplyed mask
        :param mask:  valid=1 invalid=0
        """
        if mask == None:
            return tf.reduce_mean(arr, axis=axis)
        if len(arr.shape.as_list()) < 4:
            mask = mask[:, :, :, 0]
        if arr.shape.as_list()[-1] != mask.shape.as_list()[-1]:
            mask = tf.tile(mask, (1, 1, 1, arr.shape.as_list()[-1]))
        assert arr.get_shape().as_list()[1:] == mask.get_shape().as_list()[1:]
        return tf.identity(tf.reduce_sum(arr, axis=axis) / (tf.reduce_sum(mask, axis=axis) + 1e-10), name=name)

    def _concact_features(self, conv_output):
        """
        for tensorboard visualize features
        :param conv_output:
        :return:
        """
        num_or_size_splits = conv_output.get_shape().as_list()[-1]
        each_convs = tf.split(conv_output, num_or_size_splits=num_or_size_splits, axis=3)
        colume_size = int(math.sqrt(num_or_size_splits) / 1)
        if colume_size * colume_size < num_or_size_splits:
            colume_size += 1
        row_size = math.ceil(num_or_size_splits / colume_size)
        while num_or_size_splits < colume_size * row_size:
            each_convs = tf.concat([each_convs, tf.expand_dims(tf.ones_like(each_convs[0]), axis=0)], axis=0)
            num_or_size_splits += 1
        all_concact = None
        for i in range(row_size):
            row_concact = each_convs[i * colume_size]
            for j in range(colume_size - 1):
                row_concact = tf.concat([row_concact, each_convs[i * colume_size + j + 1]], 1)
            if i == 0:
                all_concact = row_concact
            else:
                all_concact = tf.concat([all_concact, row_concact], 2)
        return all_concact
