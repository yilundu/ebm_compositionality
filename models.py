import tensorflow as tf
from tensorflow.python.platform import flags
import numpy as np
from utils import conv_block, get_weight, attention, conv_cond_concat, init_conv_weight, init_attention_weight, init_res_weight, smart_res_block, smart_res_block_optim, init_convt_weight
from utils import init_fc_weight, smart_conv_block, smart_fc_block, smart_atten_block, groupsort, smart_convt_block, swish
from data_vis import standard_transforms
from filters import stride_3

flags.DEFINE_bool('swish_act', False, 'use the swish activation for dsprites')

FLAGS = flags.FLAGS


class CubesNet(object):
    """Construct the convolutional network specified in MAML"""
    def __init__(self, num_filters=64, num_channels=3, label_size=6):

        self.channels = num_channels
        self.dim_hidden = num_filters
        self.img_size = 64
        self.label_size = label_size
        print("label_size ", self.label_size)

    def construct_weights(self, scope=''):
        weights = {}

        dtype = tf.float32
        conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)
        k = 5

        if not FLAGS.cclass:
            classes = 1
        else:
            classes = self.label_size

        with tf.variable_scope(scope):
            init_conv_weight(weights, 'c1_pre', 3, self.channels, self.dim_hidden)
            init_res_weight(weights, 'res_optim', 3, self.dim_hidden, self.dim_hidden, classes=classes)
            init_res_weight(weights, 'res_1', 3, self.dim_hidden, 2*self.dim_hidden, classes=classes)
            init_res_weight(weights, 'res_2', 3, 2*self.dim_hidden, 2*self.dim_hidden, classes=classes)
            init_res_weight(weights, 'res_3', 3, 2*self.dim_hidden, 2*self.dim_hidden, classes=classes)
            init_res_weight(weights, 'res_4', 3, 2*self.dim_hidden, 4*self.dim_hidden, classes=classes)
            # init_fc_weight(weights, 'fc_dense', 4*4*4*self.dim_hidden, 2*self.dim_hidden)
            init_fc_weight(weights, 'fc_dense', 4*self.dim_hidden, 2*self.dim_hidden)
            init_fc_weight(weights, 'fc5', 4*self.dim_hidden, 1, spec_norm=False)

        return weights

    def forward(self, inp, weights, attention_mask, reuse=False, scope='', stop_grad=False, label=None, stop_at_grad=False, stop_batch=False):

        if FLAGS.antialias:
            # Antialias the image to smooth local perturbations
            antialias = tf.tile(stride_3, (1, 1, tf.shape(inp)[3], tf.shape(inp)[3]))
            inp = tf.nn.conv2d(inp, antialias, [1, 2, 2, 1], padding='SAME')

        channels = self.channels
        if FLAGS.augment_vis:
            for transform in standard_transforms:
                inp = transform(inp)

        batch_size = tf.shape(inp)[0]

        if FLAGS.comb_mask:
            attention_mask = tf.nn.softmax(attention_mask)
            inp = tf.reshape(tf.transpose(inp, (0, 3, 1, 2)), (tf.shape(inp)[0], channels, self.img_size, self.img_size, 1))
            attention_mask = tf.reshape(attention_mask, (tf.shape(attention_mask)[0], 1, self.img_size, self.img_size, FLAGS.cond_func))
            inp = tf.reshape(tf.transpose(inp * attention_mask, (0, 4, 1, 2, 3)), (tf.shape(inp)[0] * FLAGS.cond_func, 64, 64, channels))

        weights = weights.copy()

        if not FLAGS.cclass:
            label = None

        if stop_grad:
            for k, v in weights.items():
                if type(v) == dict:
                    v = v.copy()
                    weights[k] = v
                    for k_sub, v_sub in v.items():
                        v[k_sub] = tf.stop_gradient(v_sub)
                else:
                    weights[k] = tf.stop_gradient(v)

        if FLAGS.swish_act:
            act = swish
        else:
            act = tf.nn.leaky_relu

        inp = smart_conv_block(inp, weights, reuse, 'c1_pre', use_stride=False, activation=act)

        hidden1 = smart_res_block(inp, weights, reuse, 'res_optim', adaptive=False, label=label, act=act)
        hidden2 = smart_res_block(hidden1, weights, reuse, 'res_1', stop_batch=stop_batch, adaptive=True, label=label, act=act)
        hidden3 = smart_res_block(hidden2, weights, reuse, 'res_2', stop_batch=stop_batch, adaptive=False, label=label, act=act)
        hidden4 = smart_res_block(hidden3, weights, reuse, 'res_3', adaptive=False, downsample=False, stop_batch=stop_batch, label=label, act=act)

        hidden5 = smart_res_block(hidden4, weights, reuse, 'res_4', stop_batch=stop_batch, adaptive=True, label=label, act=act)
        hidden5 = act(hidden5)
        hidden6 = tf.reduce_mean(hidden5, axis=[1, 2])
        energy = smart_fc_block(hidden6, weights, reuse, 'fc5')

        if FLAGS.comb_mask:
            energy = tf.reduce_sum(tf.reshape(energy, (batch_size, FLAGS.cond_func)), axis=1, keepdims=True)

        return energy


class CubesNetGen(object):
    """Construct the convolutional network specified in MAML"""
    def __init__(self, num_filters=64, num_channels=3, label_size=6):
        self.channels = num_channels
        self.dim_hidden = num_filters
        self.img_size = 64
        self.label_size = label_size
        print("label_size ", self.label_size)

    def construct_weights(self, scope=''):
        weights = {}

        dtype = tf.float32
        conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)
        k = 5

        if not FLAGS.cclass:
            classes = 1
        else:
            classes = self.label_size

        with tf.variable_scope(scope):
            init_fc_weight(weights, 'fc_dense', 2*self.dim_hidden, 4*4*4*self.dim_hidden)
            init_res_weight(weights, 'res_1', 3, 4*self.dim_hidden, 2*self.dim_hidden, classes=classes)
            init_res_weight(weights, 'res_2', 3, 2*self.dim_hidden, 2*self.dim_hidden, classes=classes)
            init_res_weight(weights, 'res_3', 3, 2*self.dim_hidden, self.dim_hidden, classes=classes)
            init_res_weight(weights, 'res_4', 3, self.dim_hidden, self.dim_hidden, classes=classes)
            init_conv_weight(weights, 'c4_out', 3, self.dim_hidden, self.channels)

        return weights

    def forward(self, inp, weights, reuse=False, scope='', stop_grad=False, label=None, stop_at_grad=False, stop_batch=False):

        if FLAGS.swish_act:
            act = swish
        else:
            act = tf.nn.leaky_relu

        hidden1 = act(smart_fc_block(inp, weights, reuse, 'fc_dense'))
        hidden1 = tf.reshape(hidden1, (tf.shape(inp)[0], 4, 4, 4*self.dim_hidden))

        hidden2 = smart_res_block(hidden1, weights, reuse, 'res_1', label=label, act=act, upsample=True)
        hidden3 = smart_res_block(hidden2, weights, reuse, 'res_2', adaptive=False, label=label, act=act, upsample=True)
        hidden4 = smart_res_block(hidden3, weights, reuse, 'res_3', label=label, act=act, upsample=True)
        hidden5 = smart_res_block(hidden4, weights, reuse, 'res_4', label=label, adaptive=False, act=act, upsample=True)
        output = smart_conv_block(hidden5, weights, reuse, 'c4_out', use_stride=False, activation=None)

        return output


class ResNet128(object):
    """Construct the convolutional network specified in MAML"""

    def __init__(self, num_channels=3, num_filters=64, train=False):

        self.channels = num_channels
        self.dim_hidden = num_filters
        self.dropout = train
        self.train = train

    def construct_weights(self, scope=''):
        weights = {}
        dtype = tf.float32

        classes = 1000

        with tf.variable_scope(scope):
            # First block
            init_conv_weight(weights, 'c1_pre', 3, self.channels, 64)
            init_res_weight(weights, 'res_optim', 3, 64, self.dim_hidden, classes=classes)
            init_res_weight(weights, 'res_3', 3, self.dim_hidden, 2*self.dim_hidden, classes=classes)
            init_res_weight(weights, 'res_5', 3, 2*self.dim_hidden, 4*self.dim_hidden, classes=classes)
            init_res_weight(weights, 'res_7', 3, 4*self.dim_hidden, 8*self.dim_hidden, classes=classes)
            init_res_weight(weights, 'res_9', 3, 8*self.dim_hidden, 8*self.dim_hidden, classes=classes)
            init_res_weight(weights, 'res_10', 3, 8*self.dim_hidden, 8*self.dim_hidden, classes=classes)
            init_fc_weight(weights, 'fc5', 8*self.dim_hidden , 1, spec_norm=False)


            init_attention_weight(weights, 'atten', self.dim_hidden, self.dim_hidden / 2., trainable_gamma=True)

        return weights

    def forward(self, inp, weights, reuse=False, scope='', stop_grad=False, label=None, stop_at_grad=False, stop_batch=False):
        weights = weights.copy()
        batch = tf.shape(inp)[0]

        if FLAGS.augment_vis:
            for transform in standard_transforms:
                inp = transform(inp)

        if not FLAGS.cclass:
            label = None


        if stop_grad:
            for k, v in weights.items():
                if type(v) == dict:
                    v = v.copy()
                    weights[k] = v
                    for k_sub, v_sub in v.items():
                        v[k_sub] = tf.stop_gradient(v_sub)
                else:
                    weights[k] = tf.stop_gradient(v)

        if FLAGS.swish_act:
            act = swish
        else:
            act = tf.nn.leaky_relu

        dropout = self.dropout
        train = self.train

        # Make sure gradients are modified a bit
        inp = smart_conv_block(inp, weights, reuse, 'c1_pre', use_stride=False, activation=act)
        hidden1 = smart_res_block(inp, weights, reuse, 'res_optim', label=label, dropout=dropout, train=train, downsample=True, adaptive=False)

        if FLAGS.use_attention:
            hidden1 = smart_atten_block(hidden1, weights, reuse, 'atten', stop_at_grad=stop_at_grad)

        hidden2 = smart_res_block(hidden1, weights, reuse, 'res_3', stop_batch=stop_batch, downsample=True, adaptive=True, label=label, dropout=dropout, train=train, act=act)
        hidden3 = smart_res_block(hidden2, weights, reuse, 'res_5', stop_batch=stop_batch, downsample=True, adaptive=True, label=label, dropout=dropout, train=train, act=act)
        hidden4 = smart_res_block(hidden3, weights, reuse, 'res_7', stop_batch=stop_batch, label=label, dropout=dropout, train=train, act=act, downsample=True, adaptive=True)
        hidden5 = smart_res_block(hidden4, weights, reuse, 'res_9', stop_batch=stop_batch, label=label, dropout=dropout, train=train, act=act, downsample=True, adaptive=False)
        hidden6 = smart_res_block(hidden5, weights, reuse, 'res_10', stop_batch=stop_batch, label=label, dropout=dropout, train=train, act=act, downsample=False, adaptive=False)

        if FLAGS.swish_act:
            hidden6 = act(hidden6)
        else:
            hidden6 = tf.nn.relu(hidden6)

        hidden5 = tf.reduce_sum(hidden6, [1, 2])
        hidden6 = smart_fc_block(hidden5, weights, reuse, 'fc5')
        energy = hidden6

        return energy


class CubesPredict(object):
    def __init__(self, num_channels=3, num_filters=64):

        self.channels = num_channels
        self.dim_hidden = num_filters
        self.datasource = FLAGS.datasource

    def construct_weights(self, scope=''):
        weights = {}

        dtype = tf.float32
        conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)

        classes = 1

        with tf.variable_scope(scope):
            init_conv_weight(weights, 'c1_pre', 1, self.channels, 64, spec_norm=False)
            init_conv_weight(weights, 'c1', 4, 64, self.dim_hidden, classes=classes, spec_norm=False)
            init_conv_weight(weights, 'c2', 4, self.dim_hidden, 2*self.dim_hidden, classes=classes, spec_norm=False)
            init_conv_weight(weights, 'c3', 4, 2*self.dim_hidden, 4*self.dim_hidden, classes=classes, spec_norm=False)
            init_conv_weight(weights, 'c4', 4, 4*self.dim_hidden, 4*self.dim_hidden, classes=classes, spec_norm=False)
            init_fc_weight(weights, 'fc_dense_pos', 4*self.dim_hidden, 2*self.dim_hidden, spec_norm=False)
            init_fc_weight(weights, 'fc_dense_logit', 4*self.dim_hidden, 2*self.dim_hidden, spec_norm=False)
            init_fc_weight(weights, 'fc5_pos', 2*self.dim_hidden, 2, spec_norm=False)
            init_fc_weight(weights, 'fc5_logit', 2*self.dim_hidden, 1, spec_norm=False)

        return weights

    def forward(self, inp, weights, reuse=False, scope='', stop_grad=False, label=None, **kwargs):
        channels = self.channels
        weights = weights.copy()
        inp = tf.reshape(inp, (tf.shape(inp)[0], 64, 64, self.channels))

        if FLAGS.swish_act:
            act = swish
        else:
            act = tf.nn.leaky_relu

        if stop_grad:
            for k, v in weights.items():
                if type(v) == dict:
                    v = v.copy()
                    weights[k] = v
                    for k_sub, v_sub in v.items():
                        v[k_sub] = tf.stop_gradient(v_sub)
                else:
                    weights[k] = tf.stop_gradient(v)


        h1 = smart_conv_block(inp, weights, reuse, 'c1_pre', use_stride=False, activation=act)
        h2 = smart_conv_block(h1, weights, reuse, 'c1', use_stride=True, downsample=True, label=label, extra_bias=False, activation=act)
        h3 = smart_conv_block(h2, weights, reuse, 'c2', use_stride=True, downsample=True, label=label, extra_bias=False, activation=act)
        h4 = smart_conv_block(h3, weights, reuse, 'c3', use_stride=True, downsample=True, label=label, use_scale=False, extra_bias=False, activation=act)
        h5 = smart_conv_block(h4, weights, reuse, 'c4', use_stride=True, downsample=True, label=label, use_scale=False, extra_bias=False, activation=act)

        print(h5.get_shape())
        # h5 = tf.reshape(h5, [-1, np.prod([int(dim) for dim in h5.get_shape()[1:]])])
        h5 = tf.reduce_mean(h5, axis=[1, 2])
        h6_pos = act(smart_fc_block(h5, weights, reuse, 'fc_dense_pos'))
        h6_logit = act(smart_fc_block(h5, weights, reuse, 'fc_dense_logit'))
        pos = smart_fc_block(h6_pos, weights, reuse, 'fc5_pos')
        logit = smart_fc_block(h6_logit, weights, reuse, 'fc5_logit')

        return logit, pos
