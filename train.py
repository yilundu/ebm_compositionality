import tensorflow as tf
import numpy as np
from tensorflow.python.platform import flags

from data import Cubes, CubesColor, CubesPos, CubesContinual, CubesCrossProduct, CelebA
from models import CubesNet, CubesNetGen
import os.path as osp
import os
from baselines.logger import TensorBoardOutputFormat
from utils import average_gradients, ReplayBuffer, optimistic_restore
from tqdm import tqdm
import random
from torch.utils.data import DataLoader
import time as time
from io import StringIO
from tensorflow.core.util import event_pb2
import torch
import numpy as np
from custom_adam import AdamOptimizer
from scipy.misc import imsave
import matplotlib.pyplot as plt
import scipy.ndimage
from filters import stride_3

torch.manual_seed(0)
np.random.seed(0)
tf.set_random_seed(0)

FLAGS = flags.FLAGS


# Dataset Options
flags.DEFINE_string('datasource', 'random',
    'initialization for chains, either random or default (decorruption)')
flags.DEFINE_string('dataset', 'cubes',
    'concept combination (cubes, pairs, pos, continual, color, or cross right now)')
flags.DEFINE_integer('batch_size', 16, 'Size of inputs')
flags.DEFINE_bool('single', False, 'whether to debug by training on a single image')
flags.DEFINE_integer('data_workers', 4,
    'Number of different data workers to load data in parallel')
flags.DEFINE_integer('cond_idx', 0, 'By default, train conditional models on conditioning on position')

# General Experiment Settings
flags.DEFINE_string('logdir', 'cachedir',
    'location where log of experiments will be stored')
flags.DEFINE_string('exp', 'default', 'name of experiments')
flags.DEFINE_integer('log_interval', 10, 'log outputs every so many batches')
flags.DEFINE_integer('save_interval', 1000,'save outputs every so many batches')
flags.DEFINE_integer('test_interval', 1000,'evaluate outputs every so many batches')
flags.DEFINE_integer('resume_iter', -1, 'iteration to resume training from')
flags.DEFINE_bool('train', True, 'whether to train or test')
flags.DEFINE_integer('epoch_num', 10000, 'Number of Epochs to train on')
flags.DEFINE_float('lr', 3e-4, 'Learning for training')
flags.DEFINE_integer('num_gpus', 1, 'number of gpus to train on')

# EBM Specific Experiments Settings
flags.DEFINE_float('ml_coeff', 1.0, 'Maximum Likelihood Coefficients')
flags.DEFINE_float('l2_coeff', 1.0, 'L2 Penalty training')
flags.DEFINE_bool('cclass', True, 'Whether to conditional training in models')
flags.DEFINE_bool('model_cclass', False,'use unsupervised clustering to infer fake labels')
flags.DEFINE_integer('temperature', 1, 'Temperature for energy function')
flags.DEFINE_string('objective', 'cd', 'use either contrastive divergence objective(least stable),'
                    'logsumexp(more stable)'
                    'softplus(most stable)')
flags.DEFINE_bool('zero_kl', True, 'whether to zero out the kl loss')
flags.DEFINE_float('keep_ratio', 0.05, 'Ratio of things to keep')
flags.DEFINE_bool('fft', False, 'Run all steps of model on the Fourier domain instead of image domain')
flags.DEFINE_bool('augment_vis', False, 'Augmentations on images to improve smoothness')

# Setting for MCMC sampling
flags.DEFINE_float('proj_norm', 0.0, 'Maximum change of input images')
flags.DEFINE_string('proj_norm_type', 'li', 'Either li or l2 ball projection')
flags.DEFINE_integer('num_steps', 40, 'Steps of gradient descent for training')
flags.DEFINE_float('step_lr', 100, 'Size of steps for gradient descent')
flags.DEFINE_float('attention_lr', 1e5, 'Size of steps for gradient descent')
flags.DEFINE_bool('replay_batch', True, 'Use MCMC chains initialized from a replay buffer.')
flags.DEFINE_float('noise_scale', 1.,'Relative amount of noise for MCMC')
flags.DEFINE_bool('pcd', False, 'whether to use pcd training instead')

# Architecture Settings
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets')
flags.DEFINE_bool('spec_norm', True, 'Whether to use spectral normalization on weights')
flags.DEFINE_bool('use_attention', False, 'Whether to use self attention in network')
flags.DEFINE_bool('large_model', False, 'whether to use a large model')
flags.DEFINE_bool('larger_model', False, 'Deeper ResNet32 Network')
flags.DEFINE_bool('wider_model', False, 'Wider ResNet32 Network')

# Dataset settings
flags.DEFINE_bool('mixup', False, 'whether to add mixup to training images')
flags.DEFINE_bool('augment', False, 'whether to augmentations to images')
flags.DEFINE_float('rescale', 1.0, 'Factor to rescale inputs from 0-1 box')
flags.DEFINE_integer('celeba_cond_idx', 1, 'conditioned index to select the celeba model')


# Concept combination experiments
flags.DEFINE_bool('comb_mask', False, 'condition of combinations')
flags.DEFINE_integer('cond_func', 3, 'Number of seperate conditional masks to use')
flags.DEFINE_bool('heir_mask', False, 'training a conditional model on distance on attention mask')

# Settings for antialiasing images?
flags.DEFINE_bool('antialias', False, 'whether to antialias the image before feeding it in')

# Flags for joint learning of model with other model
flags.DEFINE_bool('prelearn_model', False, 'whether to load a prelearned model')
# flags.DEFINE_string('prelearn_exp', 'cubes_continual_819_pos', 'i guess')
# flags.DEFINE_integer('prelearn_iter', 22000, 'iteration of the experiment')
flags.DEFINE_string('prelearn_exp', 'cubes_continual_914_pos', 'i guess')
flags.DEFINE_integer('prelearn_iter', 10000, 'iteration of the experiment')
flags.DEFINE_integer('prelearn_label', 2, 'number of labels for the training')

flags.DEFINE_bool('prelearn_model_shape', False, 'whether to load a prelearned model')
flags.DEFINE_string('prelearn_exp_shape', 'cubes_continual_914_pos_shape', 'i guess')
flags.DEFINE_integer('prelearn_iter_shape', 14000, 'iteration of the experiment')
flags.DEFINE_integer('prelearn_label_shape', 2, 'number of labels for the training')

# Cross product experiments settings
flags.DEFINE_bool('cond_size', False, 'condition of color ')
flags.DEFINE_bool('cond_pos', False, 'condition of position loc')
flags.DEFINE_float('ratio', 1.0, 'ratio of data to keep')
flags.DEFINE_bool('joint_baseline', False, 'use a joint baseline to train models')

FLAGS.step_lr = FLAGS.step_lr * FLAGS.rescale
FLAGS.swish_act = True

FLAGS.batch_size *= FLAGS.num_gpus

# Settings for outputting segmentation map to GPU

print("{} batch size".format(FLAGS.batch_size))


def compress_x_mod(x_mod):
    x_mod = (255 * np.clip(x_mod, 0, FLAGS.rescale) / FLAGS.rescale).astype(np.uint8)
    return x_mod


def decompress_x_mod(x_mod):
    x_mod = x_mod / 256 * FLAGS.rescale + \
        np.random.uniform(0, 1 / 256 * FLAGS.rescale, x_mod.shape)
    return x_mod


def make_image(tensor):
    """Convert an numpy representation image to Image protobuf"""
    from PIL import Image
    if len(tensor.shape) == 4:
        _, height, width, channel = tensor.shape
    elif len(tensor.shape) == 3:
        height, width, channel = tensor.shape
    elif len(tensor.shape) == 2:
        height, width = tensor.shape
        channel = 1
    tensor = tensor.astype(np.uint8)
    image = Image.fromarray(tensor)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=image_string)


def log_image(im, logger, tag, step=0):
    im = make_image(im)

    summary = [tf.Summary.Value(tag=tag, image=im)]
    summary = tf.Summary(value=summary)
    event = event_pb2.Event(summary=summary)
    event.step = step
    logger.writer.WriteEvent(event)
    logger.writer.Flush()


def rescale_im(image):
    image = np.clip(image, 0, FLAGS.rescale)
    if FLAGS.dataset == 'mnist' or FLAGS.dataset == 'dsprites':
        return (np.clip((FLAGS.rescale - image) * 256 / FLAGS.rescale, 0, 255)).astype(np.uint8)
    else:
        return (np.clip(image * 256 / FLAGS.rescale, 0, 255)).astype(np.uint8)


def train(target_vars, saver, sess, logger, dataloader, resume_iter, logdir):
    X = target_vars['X']
    X_NOISE = target_vars['X_NOISE']
    train_op = target_vars['train_op']
    energy_pos = target_vars['energy_pos']
    energy_neg = target_vars['energy_neg']
    loss_energy = target_vars['loss_energy']
    loss_ml = target_vars['loss_ml']
    loss_total = target_vars['total_loss']
    gvs = target_vars['gvs']
    x_off = target_vars['x_off']
    x_grad = target_vars['x_grad']
    x_mod = target_vars['x_mod']
    LABEL = target_vars['LABEL']
    HIER_LABEL = target_vars['HIER_LABEL']
    LABEL_POS = target_vars['LABEL_POS']
    eps = target_vars['eps_begin']
    ATTENTION_MASK = target_vars['ATTENTION_MASK']
    attention_mask = target_vars['attention_mask']
    attention_grad = target_vars['attention_grad']

    if FLAGS.prelearn_model or FLAGS.prelearn_model_shape:
        models_pretrain = target_vars['models_pretrain']

    if not FLAGS.comb_mask:
        attention_mask = tf.zeros(1)
        attention_grad = tf.zeros(1)

    if FLAGS.use_attention:
        gamma = weights['atten']['gamma']
    else:
        gamma = tf.zeros(1)


    gvs_dict = dict(gvs)

    log_output = [
        train_op,
        energy_pos,
        energy_neg,
        eps,
        loss_energy,
        loss_ml,
        loss_total,
        x_grad,
        x_off,
        x_mod,
        attention_mask,
        attention_grad,
        *gvs_dict.keys()]
    output = [train_op, x_mod]
    print("log_output ", log_output)

    replay_buffer = ReplayBuffer(10000)
    itr = resume_iter
    x_mod = None
    gd_steps = 1

    dataloader_iterator = iter(dataloader)
    best_inception = 0.0

    for epoch in range(FLAGS.epoch_num):
        for data_corrupt, data, label in dataloader:
            data_corrupt = data_corrupt_init = data_corrupt.numpy()
            data_corrupt_init = data_corrupt.copy()

            data = data.numpy()

            if FLAGS.mixup:
                idx = np.random.permutation(data.shape[0])
                lam = np.random.beta(1, 1, size=(data.shape[0], 1, 1, 1))
                data = data * lam + data[idx] * (1 - lam)

            if FLAGS.replay_batch and (x_mod is not None) and not FLAGS.joint_baseline:
                replay_buffer.add(compress_x_mod(x_mod))

                if len(replay_buffer) > FLAGS.batch_size:
                    replay_batch = replay_buffer.sample(FLAGS.batch_size)
                    replay_batch = decompress_x_mod(replay_batch)
                    replay_mask = (
                        np.random.uniform(
                            0,
                            FLAGS.rescale,
                            FLAGS.batch_size) > FLAGS.keep_ratio)
                    data_corrupt[replay_mask] = replay_batch[replay_mask]

            if FLAGS.pcd:
                if x_mod is not None:
                    data_corrupt = x_mod

            attention_mask = np.random.uniform(-1., 1., (data.shape[0], 64, 64, int(FLAGS.cond_func)))
            feed_dict = {X_NOISE: data_corrupt, X: data, ATTENTION_MASK: attention_mask}

            if FLAGS.joint_baseline:
                feed_dict[target_vars['NOISE']] = np.random.uniform(-1., 1., (data.shape[0], 128))

            if FLAGS.prelearn_model or FLAGS.prelearn_model_shape:
                _, _, labels = zip(*models_pretrain)
                labels = [LABEL, LABEL_POS] + list(labels)
                for lp, l in zip(labels, label):
                    # print("lp, l ", lp, l)
                    # print("l shape ", l.shape)
                    feed_dict[lp] = l
            else:
                label = label.numpy()
                label_init = label.copy()
                if FLAGS.cclass:
                    feed_dict[LABEL] = label
                    feed_dict[LABEL_POS] = label_init

            if FLAGS.heir_mask:
                feed_dict[HIER_LABEL] = label

            if itr % FLAGS.log_interval == 0:
                # print(feed_dict.keys())
                # print(feed_dict)
                _, e_pos, e_neg, eps, loss_e, loss_ml, loss_total, x_grad, x_off, x_mod, attention_mask, attention_grad, * \
                    grads = sess.run(log_output, feed_dict)


                kvs = {}
                kvs['e_pos'] = e_pos.mean()
                kvs['e_pos_std'] = e_pos.std()
                kvs['e_neg'] = e_neg.mean()
                kvs['e_diff'] = kvs['e_pos'] - kvs['e_neg']
                kvs['e_neg_std'] = e_neg.std()
                kvs['loss_e'] = loss_e.mean()
                kvs['loss_ml'] = loss_ml.mean()
                kvs['loss_total'] = loss_total.mean()
                kvs['x_grad'] = np.abs(x_grad).mean()
                kvs['attention_grad'] = np.abs(attention_grad).mean()
                kvs['x_off'] = x_off.mean()
                kvs['iter'] = itr

                for v, k in zip(grads, [v.name for v in gvs_dict.values()]):
                    kvs[k] = np.abs(v).max()

                string = "Obtained a total of "
                for key, value in kvs.items():
                    string += "{}: {}, ".format(key, value)

                if kvs['e_diff'] < -0.5:
                    print("Training is unstable")
                    assert False

                print(string)
                logger.writekvs(kvs)
            else:
                _, x_mod = sess.run(output, feed_dict)

            if itr % FLAGS.save_interval == 0:
                saver.save(
                    sess,
                    osp.join(
                        FLAGS.logdir,
                        FLAGS.exp,
                        'model_{}'.format(itr)))

            if itr > 30000:
                assert False

            # For some reason conditioning on position fails earlier
            # if FLAGS.cond_pos and itr > 30000:
            #     assert False

            if itr % FLAGS.test_interval == 0 and not FLAGS.joint_baseline and FLAGS.dataset != 'celeba':
                try_im = x_mod
                orig_im = data_corrupt.squeeze()
                actual_im = rescale_im(data)

                if not FLAGS.comb_mask:
                    attention_mask = np.random.uniform(-1., 1., (data.shape[0], 64, 64, int(FLAGS.cond_func)))

                orig_im = rescale_im(orig_im)
                try_im = rescale_im(try_im).squeeze()
                attention_mask = rescale_im(attention_mask)

                for i, (im, t_im, actual_im_i, attention_im) in enumerate(
                        zip(orig_im[:20], try_im[:20], actual_im, attention_mask)):
                    im, t_im, actual_im_i, attention_im = im[::-1], t_im[::-1], actual_im_i[::-1], attention_im[::-1]
                    shape = orig_im.shape[1:]
                    new_im = np.zeros((shape[0], shape[1] * (3 + FLAGS.cond_func), *shape[2:]))
                    size = shape[1]
                    new_im[:, :size] = im
                    new_im[:, size:2 * size] = t_im
                    new_im[:, 2 * size: 3 * size] = actual_im_i

                    for i in range(FLAGS.cond_func):
                        new_im[:, (3+i) * size: (4+i) * size] = np.tile(attention_im[:, :, i:i+1], (1, 1, 3))

                    log_image(
                        new_im, logger, 'train_gen_{}'.format(itr), step=i)

                test_im = x_mod

                try:
                    data_corrupt, data, label = next(dataloader_iterator)
                except BaseException:
                    dataloader_iterator = iter(dataloader)
                    data_corrupt, data, label = next(dataloader_iterator)

                data_corrupt = data_corrupt.numpy()


            itr += 1

    saver.save(sess, osp.join(FLAGS.logdir, FLAGS.exp, 'model_{}'.format(itr)))


def test(target_vars, saver, sess, logger, dataloader):
    X_NOISE = target_vars['X_NOISE']
    X = target_vars['X']
    Y = target_vars['Y']
    LABEL = target_vars['LABEL']
    x_mod = target_vars['x_mod']
    x_mod = target_vars['test_x_mod']
    energy_neg = target_vars['energy_neg']

    np.random.seed(1)
    random.seed(1)

    output = [x_mod, energy_neg]

    dataloader_iterator = iter(dataloader)
    data_corrupt, data, label = next(dataloader_iterator)
    data_corrupt, data, label = data_corrupt.numpy(), data.numpy(), label.numpy()

    orig_im = try_im = data_corrupt

    if FLAGS.cclass:
        try_im, energy_orig, energy = sess.run(
            output, {X_NOISE: orig_im, Y: label[0:1], LABEL: label})
    else:
        try_im, energy_orig, energy = sess.run(
            output, {X_NOISE: orig_im, Y: label[0:1]})

    orig_im = rescale_im(orig_im)
    try_im = rescale_im(try_im)
    actual_im = rescale_im(data)

    for i, (im, energy_i, t_im, energy, label_i, actual_im_i) in enumerate(
            zip(orig_im, energy_orig, try_im, energy, label, actual_im)):
        label_i = np.array(label_i)

        shape = im.shape[1:]
        new_im = np.zeros((shape[0], shape[1] * 3, *shape[2:]))
        size = shape[1]
        new_im[:, :size] = im
        new_im[:, size:2 * size] = t_im

        if FLAGS.cclass:
            label_i = np.where(label_i == 1)[0][0]
            if FLAGS.dataset == 'cifar10':
                log_image(new_im, logger, '{}_{:.4f}_now_{:.4f}_{}'.format(
                    i, energy_i[0], energy[0], cifar10_map[label_i]), step=i)
            else:
                log_image(
                    new_im,
                    logger,
                    '{}_{:.4f}_now_{:.4f}_{}'.format(
                        i,
                        energy_i[0],
                        energy[0],
                        label_i),
                    step=i)
        else:
            log_image(
                new_im,
                logger,
                '{}_{:.4f}_now_{:.4f}'.format(
                    i,
                    energy_i[0],
                    energy[0]),
                step=i)

    test_ims = list(try_im)
    real_ims = list(actual_im)

    for i in tqdm(range(50000 // FLAGS.batch_size + 1)):
        try:
            data_corrupt, data, label = dataloader_iterator.next()
        except BaseException:
            dataloader_iterator = iter(dataloader)
            data_corrupt, data, label = dataloader_iterator.next()

        data_corrupt, data, label = data_corrupt.numpy(), data.numpy(), label.numpy()

        if FLAGS.cclass:
            try_im, energy_orig, energy = sess.run(
                output, {X_NOISE: data_corrupt, Y: label[0:1], LABEL: label})
        else:
            try_im, energy_orig, energy = sess.run(
                output, {X_NOISE: data_corrupt, Y: label[0:1]})

        try_im = rescale_im(try_im)
        real_im = rescale_im(data)

        test_ims.extend(list(try_im))
        real_ims.extend(list(real_im))

    score, std = get_inception_score(test_ims)
    print("Inception score of {} with std of {}".format(score, std))


def main():

    logdir = osp.join(FLAGS.logdir, FLAGS.exp)
    logger = TensorBoardOutputFormat(logdir)

    config = tf.ConfigProto()

    sess = tf.Session(config=config)
    LABEL = None
    print("Loading data...")
    if FLAGS.dataset == 'cubes':
        dataset = Cubes(cond_idx=FLAGS.cond_idx)
        test_dataset = dataset

        if FLAGS.cond_idx == 0:
            label_size = 2
        elif FLAGS.cond_idx == 1:
            label_size = 1
        elif FLAGS.cond_idx == 2:
            label_size = 3
        elif FLAGS.cond_idx == 3:
            label_size = 20

        LABEL = tf.placeholder(shape=(None, label_size), dtype=tf.float32)
        LABEL_POS = tf.placeholder(shape=(None, label_size), dtype=tf.float32)
    elif FLAGS.dataset == 'color':
        dataset = CubesColor()
        test_dataset = dataset
        LABEL = tf.placeholder(shape=(None, 301), dtype=tf.float32)
        LABEL_POS = tf.placeholder(shape=(None, 301), dtype=tf.float32)
        label_size = 301
    elif FLAGS.dataset == 'pos':
        dataset = CubesPos()
        test_dataset = dataset
        LABEL = tf.placeholder(shape=(None, 2), dtype=tf.float32)
        LABEL_POS = tf.placeholder(shape=(None, 2), dtype=tf.float32)
        label_size = 2
    elif FLAGS.dataset == "pairs":
        dataset = Pairs(cond_idx=0)
        test_dataset = dataset
        LABEL = tf.placeholder(shape=(None, 6), dtype=tf.float32)
        LABEL_POS = tf.placeholder(shape=(None, 6), dtype=tf.float32)
        label_size = 6
    elif FLAGS.dataset == "continual":
        dataset = CubesContinual()
        test_dataset = dataset

        if FLAGS.prelearn_model_shape:
            LABEL = tf.placeholder(shape=(None, 20), dtype=tf.float32)
            LABEL_POS = tf.placeholder(shape=(None, 20), dtype=tf.float32)
            label_size = 20
        else:
            LABEL = tf.placeholder(shape=(None, 2), dtype=tf.float32)
            LABEL_POS = tf.placeholder(shape=(None, 2), dtype=tf.float32)
            label_size = 2

    elif FLAGS.dataset == "cross":
        dataset = CubesCrossProduct(FLAGS.ratio, cond_size=FLAGS.cond_size, cond_pos=FLAGS.cond_pos, joint_baseline=FLAGS.joint_baseline)
        test_dataset = dataset

        if FLAGS.cond_size:
            LABEL = tf.placeholder(shape=(None, 1), dtype=tf.float32)
            LABEL_POS = tf.placeholder(shape=(None, 1), dtype=tf.float32)
            label_size = 1
        elif FLAGS.cond_pos:
            LABEL = tf.placeholder(shape=(None, 2), dtype=tf.float32)
            LABEL_POS = tf.placeholder(shape=(None, 2), dtype=tf.float32)
            label_size = 2

        if FLAGS.joint_baseline:
            LABEL = tf.placeholder(shape=(None, 3), dtype=tf.float32)
            LABEL_POS = tf.placeholder(shape=(None, 3), dtype=tf.float32)
            label_size = 3

    elif FLAGS.dataset == 'celeba':
        dataset = CelebA(cond_idx=FLAGS.celeba_cond_idx)
        test_dataset = dataset
        channel_num = 3
        X_NOISE = tf.placeholder(shape=(None, 128, 128, 3), dtype=tf.float32)
        X = tf.placeholder(shape=(None, 128, 128, 3), dtype=tf.float32)
        LABEL = tf.placeholder(shape=(None, 2), dtype=tf.float32)
        LABEL_POS = tf.placeholder(shape=(None, 2), dtype=tf.float32)

        model = ResNet128(
            num_channels=channel_num,
            num_filters=64,
            classes=2)

    if FLAGS.joint_baseline:
        # Other stuff for joint model
        optimizer = AdamOptimizer(FLAGS.lr, beta1=0.99, beta2=0.999)

        X = tf.placeholder(shape=(None, 64, 64, 3), dtype=tf.float32)
        X_NOISE = tf.placeholder(shape=(None, 64, 64, 3), dtype=tf.float32)
        ATTENTION_MASK = tf.placeholder(shape=(None, 64, 64, FLAGS.cond_func), dtype=tf.float32)
        NOISE = tf.placeholder(shape=(None, 128), dtype=tf.float32)
        HIER_LABEL = tf.placeholder(shape=(None, 2), dtype=tf.float32)

        channel_num = 3

        model = CubesNetGen(num_channels=channel_num, label_size=label_size)
        weights = model.construct_weights('context_0')
        output = model.forward(NOISE, weights, reuse=False, label=LABEL)
        print(output.get_shape())
        mse_loss = tf.reduce_mean(tf.square(output - X))
        gvs = optimizer.compute_gradients(mse_loss)
        train_op = optimizer.apply_gradients(gvs)
        gvs = [(k, v) for (k, v) in gvs if k is not None]

        target_vars = {}
        target_vars['train_op'] = train_op
        target_vars['X'] = X
        target_vars['X_NOISE'] = X_NOISE
        target_vars['ATTENTION_MASK'] = ATTENTION_MASK
        target_vars['eps_begin'] = tf.zeros(1)
        target_vars['gvs'] = gvs
        target_vars['energy_pos'] = tf.zeros(1)
        target_vars['energy_neg'] = tf.zeros(1)
        target_vars['loss_energy'] = tf.zeros(1)
        target_vars['loss_ml'] = tf.zeros(1)
        target_vars['total_loss'] = mse_loss
        target_vars['attention_mask'] = tf.zeros(1)
        target_vars['attention_grad'] = tf.zeros(1)
        target_vars['x_off'] = tf.reduce_mean(tf.abs(output - X))
        target_vars['x_mod'] = tf.zeros(1)
        target_vars['x_grad'] = tf.zeros(1)
        target_vars['NOISE'] = NOISE
        target_vars['LABEL'] = LABEL
        target_vars['LABEL_POS'] = LABEL_POS
        target_vars['HIER_LABEL'] = HIER_LABEL

        data_loader = DataLoader(
            dataset,
            batch_size=FLAGS.batch_size,
            num_workers=FLAGS.data_workers,
            drop_last=True,
            shuffle=True)
    else:
        print("label size here ", label_size)
        channel_num = 3
        X_NOISE = tf.placeholder(shape=(None, 64, 64, 3), dtype=tf.float32)
        X = tf.placeholder(shape=(None, 64, 64, 3), dtype=tf.float32)
        HEIR_LABEL = tf.placeholder(shape=(None, 2), dtype=tf.float32)
        ATTENTION_MASK = tf.placeholder(shape=(None, 64, 64, FLAGS.cond_func), dtype=tf.float32)

        if FLAGS.dataset != "celeba":
            model = CubesNet(num_channels=channel_num, label_size=label_size)

        heir_model = HeirNet(num_channels=FLAGS.cond_func)

        models_pretrain = []
        if FLAGS.prelearn_model:
            model_prelearn = CubesNet(num_channels=channel_num, label_size=FLAGS.prelearn_label)
            weights = model_prelearn.construct_weights('context_1')
            LABEL_PRELEARN = tf.placeholder(shape=(None, FLAGS.prelearn_label), dtype=tf.float32)
            models_pretrain.append((model_prelearn, weights, LABEL_PRELEARN))

            cubes_logdir = osp.join(FLAGS.logdir, FLAGS.prelearn_exp)
            if (FLAGS.prelearn_iter != -1 or not FLAGS.train):
                model_file = osp.join(cubes_logdir, 'model_{}'.format(FLAGS.prelearn_iter))
                resume_itr = FLAGS.resume_iter
                # saver.restore(sess, model_file)

                v_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='context_{}'.format(1))
                v_map = {(v.name.replace('context_{}'.format(1), 'context_0')[:-2]): v for v in v_list}
                saver = tf.train.Saver(v_map)
                saver.restore(sess, model_file)

        if FLAGS.prelearn_model_shape:
            model_prelearn = CubesNet(num_channels=channel_num, label_size=FLAGS.prelearn_label_shape)
            weights = model_prelearn.construct_weights('context_2')
            LABEL_PRELEARN = tf.placeholder(shape=(None, FLAGS.prelearn_label_shape), dtype=tf.float32)
            models_pretrain.append((model_prelearn, weights, LABEL_PRELEARN))

            cubes_logdir = osp.join(FLAGS.logdir, FLAGS.prelearn_exp_shape)
            if (FLAGS.prelearn_iter_shape != -1 or not FLAGS.train):
                model_file = osp.join(cubes_logdir, 'model_{}'.format(FLAGS.prelearn_iter_shape))
                resume_itr = FLAGS.resume_iter
                # saver.restore(sess, model_file)

                v_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='context_{}'.format(2))
                v_map = {(v.name.replace('context_{}'.format(2), 'context_0')[:-2]): v for v in v_list}
                saver = tf.train.Saver(v_map)
                saver.restore(sess, model_file)

        print("Done loading...")

        data_loader = DataLoader(
            dataset,
            batch_size=FLAGS.batch_size,
            num_workers=FLAGS.data_workers,
            drop_last=True,
            shuffle=True)

        batch_size = FLAGS.batch_size

        weights = model.construct_weights('context_0')

        if FLAGS.heir_mask:
            weights = heir_model.construct_weights('heir_0', weights=weights)

        Y = tf.placeholder(shape=(None), dtype=tf.int32)

        # Varibles to run in training

        X_SPLIT = tf.split(X, FLAGS.num_gpus)
        X_NOISE_SPLIT = tf.split(X_NOISE, FLAGS.num_gpus)
        LABEL_SPLIT = tf.split(LABEL, FLAGS.num_gpus)
        LABEL_POS_SPLIT = tf.split(LABEL_POS, FLAGS.num_gpus)
        LABEL_SPLIT_INIT = list(LABEL_SPLIT)
        attention_mask = ATTENTION_MASK
        tower_grads = []
        tower_gen_grads = []
        x_mod_list = []

        optimizer = AdamOptimizer(FLAGS.lr, beta1=0.0, beta2=0.99)

        for j in range(FLAGS.num_gpus):

            x_mod = X_SPLIT[j]
            if FLAGS.comb_mask:
                steps = tf.constant(0)
                c = lambda i, x: tf.less(i, FLAGS.num_steps)

                def langevin_attention_step(counter, attention_mask):
                    attention_mask = attention_mask + tf.random_normal(tf.shape(attention_mask), mean=0.0, stddev=0.01)
                    energy_noise = energy_start = model.forward(
                                x_mod,
                                weights,
                                attention_mask,
                                label=LABEL_SPLIT[j],
                                reuse=True,
                                stop_at_grad=False,
                                stop_batch=True)

                    if FLAGS.heir_mask:
                        energy_heir = 1.00 * heir_model.forward(attention_mask, weights, label=HEIR_LABEL)
                        energy_noise = energy_noise + energy_heir

                    attention_grad = tf.gradients(
                        FLAGS.temperature * energy_noise, [attention_mask])[0]
                    energy_noise_old = energy_noise

                    # Clip gradient norm for now
                    attention_mask = attention_mask - (FLAGS.attention_lr) * attention_grad
                    attention_mask = tf.layers.average_pooling2d(attention_mask, (3, 3), 1, padding='SAME')
                    attention_mask = tf.stop_gradient(attention_mask)

                    counter = counter + 1

                    return counter, attention_mask

                steps, attention_mask = tf.while_loop(c, langevin_attention_step, (steps, attention_mask))

                # attention_mask = tf.Print(attention_mask, [attention_mask])

                energy_pos = model.forward(
                        X_SPLIT[j],
                        weights,
                        tf.stop_gradient(attention_mask),
                        label=LABEL_POS_SPLIT[j],
                        stop_at_grad=False)

                if FLAGS.heir_mask:
                    energy_heir = 1.00 * heir_model.forward(attention_mask, weights, label=HEIR_LABEL)
                    energy_pos = energy_heir + energy_pos

            else:
                energy_pos = model.forward(
                        X_SPLIT[j],
                        weights,
                        attention_mask,
                        label=LABEL_POS_SPLIT[j],
                        stop_at_grad=False)

                if FLAGS.heir_mask:
                    energy_heir = 1.00 * heir_model.forward(attention_mask, weights, label=HEIR_LABEL)
                    energy_pos = energy_heir + energy_pos

            print("Building graph...")
            x_mod = x_orig = X_NOISE_SPLIT[j]

            x_grads = []

            loss_energys = []

            eps_begin = tf.zeros(1)

            steps = tf.constant(0)
            c_cond = lambda i, x, y: tf.less(i, FLAGS.num_steps)

            def langevin_step(counter, x_mod, attention_mask):

                lr = FLAGS.step_lr

                x_mod = x_mod + tf.random_normal(tf.shape(x_mod),
                                                 mean=0.0,
                                                 stddev=0.001 * FLAGS.rescale * FLAGS.noise_scale)
                attention_mask = attention_mask + tf.random_normal(tf.shape(attention_mask), mean=0.0, stddev=0.01)

                energy_noise = model.forward(
                            x_mod,
                            weights,
                            attention_mask,
                            label=LABEL_SPLIT[j],
                            reuse=True,
                            stop_at_grad=False,
                            stop_batch=True)

                if FLAGS.prelearn_model:
                    for m_i, w_i, l_i in models_pretrain:
                        energy_noise = energy_noise + m_i.forward(
                                    x_mod,
                                    w_i,
                                    attention_mask,
                                    label=l_i,
                                    reuse=True,
                                    stop_at_grad=False,
                                    stop_batch=True)


                if FLAGS.heir_mask:
                    energy_heir = 1.00 * heir_model.forward(attention_mask, weights, label=HEIR_LABEL)
                    energy_noise = energy_heir + energy_noise

                x_grad, attention_grad = tf.gradients(
                    FLAGS.temperature * energy_noise, [x_mod, attention_mask])

                if not FLAGS.comb_mask:
                    attention_grad = tf.zeros(1)
                energy_noise_old = energy_noise

                if FLAGS.proj_norm != 0.0:
                    if FLAGS.proj_norm_type == 'l2':
                        x_grad = tf.clip_by_norm(x_grad, FLAGS.proj_norm)
                    elif FLAGS.proj_norm_type == 'li':
                        x_grad = tf.clip_by_value(
                            x_grad, -FLAGS.proj_norm, FLAGS.proj_norm)
                    else:
                        print("Other types of projection are not supported!!!")
                        assert False

                # Clip gradient norm for now
                x_last = x_mod - (lr) * x_grad

                if FLAGS.comb_mask:
                    attention_mask = attention_mask - FLAGS.attention_lr * attention_grad
                    attention_mask = tf.layers.average_pooling2d(attention_mask, (3, 3), 1, padding='SAME')
                    attention_mask = tf.stop_gradient(attention_mask)

                x_mod = x_last
                x_mod = tf.clip_by_value(x_mod, 0, FLAGS.rescale)

                counter = counter + 1

                return counter, x_mod, attention_mask


            steps, x_mod, attention_mask = tf.while_loop(c_cond, langevin_step, (steps, x_mod, attention_mask))

            attention_mask = tf.stop_gradient(attention_mask)
            # attention_mask = tf.Print(attention_mask, [attention_mask])

            energy_eval = model.forward(x_mod, weights, attention_mask, label=LABEL_SPLIT[j],
                                        stop_at_grad=False, reuse=True)
            x_grad, attention_grad = tf.gradients(FLAGS.temperature * energy_eval, [x_mod, attention_mask])
            x_grads.append(x_grad)

            energy_neg = model.forward(
                    tf.stop_gradient(x_mod),
                    weights,
                    tf.stop_gradient(attention_mask),
                    label=LABEL_SPLIT[j],
                    stop_at_grad=False,
                    reuse=True)

            if FLAGS.heir_mask:
                energy_heir = 1.00 * heir_model.forward(attention_mask, weights, label=HEIR_LABEL)
                energy_neg = energy_heir + energy_neg


            temp = FLAGS.temperature

            x_off = tf.reduce_mean(
                tf.abs(x_mod[:tf.shape(X_SPLIT[j])[0]] - X_SPLIT[j]))

            loss_energy = model.forward(
                x_mod,
                weights,
                attention_mask,
                reuse=True,
                label=LABEL,
                stop_grad=True)

            print("Finished processing loop construction ...")

            target_vars = {}

            if FLAGS.antialias:
                antialias = tf.tile(stride_3, (1, 1, tf.shape(x_mod)[3], tf.shape(x_mod)[3]))
                inp = tf.nn.conv2d(x_mod, antialias, [1, 2, 2, 1], padding='SAME')

            test_x_mod = x_mod

            if FLAGS.cclass or FLAGS.model_cclass:
                label_sum = tf.reduce_sum(LABEL_SPLIT[0], axis=0)
                label_prob = label_sum / tf.reduce_sum(label_sum)
                label_ent = -tf.reduce_sum(label_prob *
                                           tf.math.log(label_prob + 1e-7))
            else:
                label_ent = tf.zeros(1)

            target_vars['label_ent'] = label_ent

            if FLAGS.train:
                if FLAGS.objective == 'logsumexp':
                    pos_term = temp * energy_pos
                    energy_neg_reduced = (energy_neg - tf.reduce_min(energy_neg))
                    coeff = tf.stop_gradient(tf.exp(-temp * energy_neg_reduced))
                    norm_constant = tf.stop_gradient(tf.reduce_sum(coeff)) + 1e-4
                    pos_loss = tf.reduce_mean(temp * energy_pos)
                    neg_loss = coeff * (-1 * temp * energy_neg) / norm_constant
                    loss_ml = FLAGS.ml_coeff * (pos_loss + tf.reduce_sum(neg_loss))
                elif FLAGS.objective == 'cd':
                    pos_loss = tf.reduce_mean(temp * energy_pos)
                    neg_loss = -tf.reduce_mean(temp * energy_neg)
                    loss_ml = FLAGS.ml_coeff * (pos_loss + tf.reduce_sum(neg_loss))
                elif FLAGS.objective == 'softplus':
                    loss_ml = FLAGS.ml_coeff * \
                        tf.nn.softplus(temp * (energy_pos - energy_neg))

                loss_total = tf.reduce_mean(loss_ml)

                if not FLAGS.zero_kl:
                    loss_total = loss_total + tf.reduce_mean(loss_energy)

                loss_total = loss_total + \
                    FLAGS.l2_coeff * (tf.reduce_mean(tf.square(energy_pos)) + tf.reduce_mean(tf.square((energy_neg))))

                print("Started gradient computation...")
                gvs = optimizer.compute_gradients(loss_total)
                gvs = [(k, v) for (k, v) in gvs if k is not None]

                print("Applying gradients...")

                tower_grads.append(gvs)

                print("Finished applying gradients.")

                target_vars['loss_ml'] = loss_ml
                target_vars['total_loss'] = loss_total
                target_vars['loss_energy'] = loss_energy
                target_vars['weights'] = weights
                target_vars['gvs'] = gvs

            target_vars['X'] = X
            target_vars['Y'] = Y
            target_vars['LABEL'] = LABEL
            target_vars['HIER_LABEL'] = HEIR_LABEL
            target_vars['LABEL_POS'] = LABEL_POS
            target_vars['X_NOISE'] = X_NOISE
            target_vars['energy_pos'] = energy_pos
            target_vars['attention_grad'] = attention_grad

            if len(x_grads) >= 1:
                target_vars['x_grad'] = x_grads[-1]
                target_vars['x_grad_first'] = x_grads[0]
            else:
                target_vars['x_grad'] = tf.zeros(1)
                target_vars['x_grad_first'] = tf.zeros(1)

            target_vars['x_mod'] = x_mod
            target_vars['x_off'] = x_off
            target_vars['temp'] = temp
            target_vars['energy_neg'] = energy_neg
            target_vars['test_x_mod'] = test_x_mod
            target_vars['eps_begin'] = eps_begin
            target_vars['ATTENTION_MASK'] = ATTENTION_MASK
            target_vars['models_pretrain'] = models_pretrain
            if FLAGS.comb_mask:
                target_vars['attention_mask'] = tf.nn.softmax(attention_mask)
            else:
                target_vars['attention_mask'] = tf.zeros(1)

        if FLAGS.train:
            grads = average_gradients(tower_grads)
            train_op = optimizer.apply_gradients(grads)
            target_vars['train_op'] = train_op

    # sess = tf.Session(config=config)

    saver = loader = tf.train.Saver(
        max_to_keep=30, keep_checkpoint_every_n_hours=6)

    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("Model has a total of {} parameters".format(total_parameters))

    sess.run(tf.global_variables_initializer())

    resume_itr = 0

    if (FLAGS.resume_iter != -1 or not FLAGS.train):
        model_file = osp.join(logdir, 'model_{}'.format(FLAGS.resume_iter))
        resume_itr = FLAGS.resume_iter
        # saver.restore(sess, model_file)
        optimistic_restore(sess, model_file)

    print("Initializing variables...")

    print("Start broadcast")
    print("End broadcast")

    if FLAGS.train:
        train(target_vars, saver, sess,
              logger, data_loader, resume_itr,
              logdir)

    test(target_vars, saver, sess, logger, data_loader)


if __name__ == "__main__":
    main()
