import tensorflow as tf
import math
from tqdm import tqdm
from tensorflow.python.platform import flags
from torch.utils.data import DataLoader, Dataset
from models import ResNet128
from utils import optimistic_restore, ReplayBuffer
import os.path as osp
import numpy as np
from baselines.logger import TensorBoardOutputFormat
from scipy.misc import imsave
import os
from custom_adam import AdamOptimizer
from itertools import product

flags.DEFINE_integer('batch_size', 256, 'Size of inputs')
flags.DEFINE_integer('data_workers', 4, 'Number of workers to do things')
flags.DEFINE_string('logdir', 'cachedir', 'directory for logging')
flags.DEFINE_string('savedir', 'cachedir', 'location where log of experiments will be stored')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omniglot.')
flags.DEFINE_float('step_lr', 100.0, 'size of gradient descent size')
flags.DEFINE_bool('cclass', True, 'not cclass')
flags.DEFINE_bool('proj_cclass', False, 'use for backwards compatibility reasons')
flags.DEFINE_bool('spec_norm', True, 'Whether to use spectral normalization on weights')
flags.DEFINE_bool('use_bias', True, 'Whether to use bias in convolution')
flags.DEFINE_bool('use_attention', False, 'Whether to use self attention in network')
flags.DEFINE_integer('num_steps', 100, 'number of steps to optimize the label')
flags.DEFINE_string('task', 'negation_figure', 'conceptcombine, combination_figure, negation_figure, or_figure')

flags.DEFINE_bool('latent_energy', False, 'latent energy in model')
flags.DEFINE_bool('proj_latent', False, 'Projection of latents')


# Whether to train for gentest
flags.DEFINE_bool('train', False, 'whether to train on generalization into multiple different predictions')

FLAGS = flags.FLAGS
FLAGS.swish_act = True


def conceptcombine(sess, kvs, save_exp_dir):
    n = 5

    labels = kvs['labels']
    x_mod = kvs['x_mod']
    X_NOISE = kvs['X_NOISE']
    model_base = kvs['model_base']
    weights = kvs['weights']

    factors = len(labels)
    factors = 2
    prod_labels = np.array(list(product(*[[0, 1] for i in range(factors)])))
    print(prod_labels)
    prod_labels = np.reshape(np.tile(prod_labels[:, None, :], (1, n, 1)), (-1, 2))

    feed_dict = {}

    for i, label in enumerate(labels):
        feed_dict[label] = np.eye(2)[prod_labels[:, i]]

    x_noise = np.random.uniform(0, 1, (prod_labels.shape[0], 128, 128, 3))
    feed_dict[X_NOISE] = x_noise

    output = sess.run([x_mod], feed_dict)[0]
    output = output.reshape((-1, 5, 128, 128, 3)).transpose((0, 2, 1, 3, 4)).reshape((-1, 128 * n, 3))
    print(output.shape, output.max(), output.min())
    imsave("debug.png", output)


def combination_figure(sess, kvs, select_idx):
    n = 16

    labels = kvs['labels']
    x_mod = kvs['x_mod']
    X_NOISE = kvs['X_NOISE']
    model_base = kvs['model_base']
    weights = kvs['weights']
    feed_dict = {}

    for i, label in enumerate(labels):
        j = select_idx[i]
        feed_dict[label] = np.tile(np.eye(2)[j:j+1], (16, 1))

    x_noise = np.random.uniform(0, 1, (n, 128, 128, 3))
    feed_dict[X_NOISE] = x_noise

    output = sess.run([x_mod], feed_dict)[0]
    output = output.reshape((n * 128, 128, 3))
    imsave("debug.png", output)


def negation_figure(sess, kvs, select_idx):
    n = 16

    labels = kvs['labels']
    x_mod = kvs['x_mod']
    X_NOISE = kvs['X_NOISE']
    model_base = kvs['model_base']
    weights = kvs['weights']
    feed_dict = {}

    for i, label in enumerate(labels):
        j = select_idx[i]
        feed_dict[label] = np.tile(np.eye(2)[j:j+1], (16, 1))

    x_noise = np.random.uniform(0, 1, (n, 128, 128, 3))
    feed_dict[X_NOISE] = x_noise

    output = sess.run([x_mod], feed_dict)[0]
    output = output.reshape((n * 128, 128, 3))
    imsave("debug.png", output)


def combine_main(models, resume_iters, select_idx):
    config = tf.ConfigProto()
    sess = tf.Session(config=config)

    weights = []
    model_base = ResNet128(classes=2, num_filters=64)
    labels = []

    for i, (model_name, resume_iter) in enumerate(zip(models, resume_iters)):
        # Model 1 will be conditioned on size
        weight = model_base.construct_weights('context_{}'.format(i))
        weights.append(weight)
        LABEL_COND = tf.placeholder(shape=(None, 2), dtype=tf.float32)
        labels.append(LABEL_COND)

    # Finish initializing all variables
    sess.run(tf.global_variables_initializer())

    # Now go load the correct files
    for i, (model_name, resume_iter) in enumerate(zip(models, resume_iters)):
        # Model 1 will be conditioned on size
        save_path_size = osp.join(FLAGS.logdir, model_name, 'model_{}'.format(resume_iter))
        v_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='context_{}'.format(i))
        v_map = {(v.name.replace('context_{}'.format(i), 'context_0')[:-2]): v for v in v_list}
        saver = tf.train.Saver(v_map)
        saver.restore(sess, save_path_size)

    X_NOISE = tf.placeholder(shape=(None, 128, 128, 3), dtype=tf.float32)

    c = lambda i, x: tf.less(i, FLAGS.num_steps)
    steps = tf.constant(0)

    task = FLAGS.task
    def langevin_step(counter, x_mod):
        x_mod = x_mod + tf.random_normal(tf.shape(x_mod), mean=0.0, stddev=0.005)

        if task == 'or_figure':
            e1 = model_base.forward(x_mod, weights[0], label=labels[0]) + model_base.forward(x_mod, weights[1], label=labels[1])
            e2 = model_base.forward(x_mod, weights[2], label=labels[2]) + model_base.forward(x_mod, weights[3], label=labels[3])
            scale = 100

            e_pos = -tf.math.reduce_logsumexp(scale * tf.concat([-e1, -e2], axis=1), axis=1, keepdims=True) / scale
            e_pos = e2
        else:
            e_pos = 0
            for i, (weight, label) in enumerate(zip(weights, labels)):
                print(i)
                if i > 0 and task == 'negation_figure':
                    if i == 1:
                        e_pos = -0.001 * model_base.forward(x_mod, weight, label=label) + e_pos
                    if i == 2:
                        e_pos = -0.001 * model_base.forward(x_mod, weight, label=label) + e_pos
                else:
                    e_pos = model_base.forward(x_mod, weight, label=label) + e_pos

        x_grad = tf.gradients(e_pos, [x_mod])[0]
        x_mod = x_mod - FLAGS.step_lr * x_grad
        x_mod = tf.clip_by_value(x_mod, 0, 1)

        counter = counter + 1

        return counter, x_mod

    steps, x_mod = tf.while_loop(c, langevin_step, (steps, X_NOISE))

    kvs = {}
    kvs['X_NOISE'] = X_NOISE
    kvs['x_mod'] = x_mod
    kvs['labels'] = labels
    kvs['model_base'] = model_base
    kvs['weights'] = weights

    if FLAGS.task == 'conceptcombine':
        save_exp_dir = osp.join(FLAGS.savedir, '{}_{}_joint'.format(*models))

        if not osp.exists(save_exp_dir):
            os.makedirs(save_exp_dir)

        print("Saving models at {}".format(save_exp_dir))

        conceptcombine(sess, kvs, save_exp_dir)
    elif FLAGS.task == 'combination_figure':
        combination_figure(sess, kvs, select_idx)
    elif FLAGS.task == 'negation_figure':
        negation_figure(sess, kvs, select_idx)
    elif FLAGS.task == 'or_figure':
        negation_figure(sess, kvs, select_idx)


if __name__ == "__main__":
    models_orig = ['celeba_smiling', 'celeba_male', 'celeba_attractive', 'celeba_black', 'celeba_old', 'celeba_wavy_hair', 'celeba_old']
    resume_iters_orig = [24000, 23000, 22000, 32000, 24000, 24000, 24000]


    ##################################
    # Settings for the composition_figure
    models = [models_orig[6]]
    resume_iters = [resume_iters_orig[6]]
    select_idx = [1]

    models = models + [models_orig[1]]
    resume_iters = resume_iters + [resume_iters_orig[1]]
    select_idx = select_idx + [0]

    models = models + [models_orig[0]]
    resume_iters = resume_iters + [resume_iters_orig[0]]
    select_idx = select_idx + [1]

    models = models + [models_orig[5]]
    resume_iters = resume_iters + [resume_iters_orig[5]]
    select_idx = select_idx + [1]

    combine_main(models, resume_iters, select_idx)

    # List of 4 attributes that might be good
    # Young -> Female -> Smiling -> Wavy

    #################################
    # Settings for the negation_figure

    # models = []
    # resume_iters = []
    # select_idx = []
    # # By default the second model will be the negated model
    # models = [models_orig[0]]
    # resume_iters = [resume_iters_orig[0]]
    # select_idx = [1]

    # models = models + [models_orig[1]]
    # resume_iters = resume_iters + [resume_iters_orig[1]]
    # select_idx = select_idx + [0]

    # combine_main(models, resume_iters, select_idx)

    #################################
    # Settings for the or figure

    # models = []
    # resume_iters = []
    # select_idx = []
    # # By default the second model will be the negated model
    # models = [models_orig[0], models_orig[1]]
    # resume_iters = [resume_iters_orig[0],  resume_iters_orig[1]]
    # select_idx = [1, 1]

    # models = models + [models_orig[0], models_orig[1]]
    # resume_iters = resume_iters + [resume_iters_orig[0], resume_iters_orig[1]]
    # select_idx = select_idx + [0, 0]

    # combine_main(models, resume_iters, select_idx)

    #################################
    # Settings for the Venn Diagram Figure
    # models = []
    # resume_iters = []
    # select_idx = []
    # # By default the second model will be the negated model

    # models = models + [models_orig[3]]
    # resume_iters = resume_iters + [resume_iters_orig[3]]
    # select_idx = select_idx + [1]

    # models = models + [models_orig[0]]
    # resume_iters = resume_iters + [resume_iters_orig[0]]
    # select_idx = select_idx + [1]

    # models = models + [models_orig[1]]
    # resume_iters = resume_iters + [resume_iters_orig[1]]
    # select_idx = select_idx + [1]


    # combine_main(models, resume_iters, select_idx)
