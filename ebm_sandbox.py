import tensorflow as tf
import math
from tqdm import tqdm
from tensorflow.python.platform import flags
from torch.utils.data import DataLoader
import torch
from models import CubesNet, HeirNet, CubesNetGen, CubesPredict
from data import Cubes, CubesCrossProduct
from utils import optimistic_restore, set_seed
import os.path as osp
import numpy as np
from baselines.logger import TensorBoardOutputFormat
from scipy.misc import imsave
import os
import sklearn.metrics as sk
from baselines.common.tf_util import initialize
from scipy.linalg import eig
import matplotlib.pyplot as plt
from data import CubesColor, CubesPos, Cubes
from custom_adam import AdamOptimizer
from scipy import ndimage

set_seed(2)
# set_seed(3)
# set_seed(4)

flags.DEFINE_string('datasource', 'random', 'default or noise or negative or single')
flags.DEFINE_string('dataset', 'cifar10', 'omniglot or imagenet or omniglotfull or cifar10 or mnist or dsprites')
flags.DEFINE_string('logdir', 'cachedir', 'location where log of experiments will be stored')
flags.DEFINE_string('task', 'concept_combine', 'concept_combine where two different EBMs are combined to represent new combinations not seen before'
                    'multi for combining multiple different EBMs'
                    'continual for continual learning different EBMs'
                    'infer_pos infer positions based off an ebm'
                    'cross_benchmark run a cross benchmark of results'
                    'composition_figure for visualizing compositions of concepts'
                    )
flags.DEFINE_string('exp', 'default', 'name of experiments')
flags.DEFINE_integer('data_workers', 5, 'Number of different data workers to load data in parallel')
flags.DEFINE_integer('batch_size', 64, 'Size of inputs')
flags.DEFINE_integer('resume_iter', -1, 'iteration to resume training from')
flags.DEFINE_bool('bn', False, 'Whether to use batch normalization or not')
flags.DEFINE_bool('spec_norm', True, 'Whether to use spectral normalization on weights')
flags.DEFINE_bool('train', True, 'Whether to train or test network')
flags.DEFINE_bool('single', False, 'whether to use one sample to debug')
flags.DEFINE_bool('cclass', True, 'whether to use a conditional model (required for task label)')
flags.DEFINE_integer('num_steps', 200, 'number of steps to optimize the label')
flags.DEFINE_float('step_lr', 100.0, 'step size for updates on label')
flags.DEFINE_float('proj_norm', 0.0, 'Maximum change of input images')

# Heirarchy specific hyperparameters
flags.DEFINE_integer('cond_func', 3, 'number of masks to condition by')
flags.DEFINE_bool('heir_mask', False, 'training a conditional model on distance on attention mask')
flags.DEFINE_bool('antialias', False, 'whether to antialias the image before feeding it in')
flags.DEFINE_bool('augment_vis', False, 'augment visualization')
flags.DEFINE_bool('comb_mask', False, 'don"t you comb_mask"')

# Experiment with pos and color
flags.DEFINE_string('exp_color', '818_color_different', 'name of experiments')
flags.DEFINE_string('exp_pos', '818_pos_cond', 'name of experiments')
flags.DEFINE_integer('resume_pos', 10000, 'Second iteration to resume')
flags.DEFINE_integer('resume_color', 31000, 'Second iteration to resume')

# Experiment with continual learning of position and shape
flags.DEFINE_string('continual_pos', 'cubes_continual_914_pos', 'name of experiments')
flags.DEFINE_string('continual_shape', 'cubes_continual_914_pos_shape', 'name of experiments')
flags.DEFINE_string('continual_color', 'cubes_continual_914_pos_shape_color', 'name of experiments')
flags.DEFINE_integer('continual_resume_pos', 10000, 'Second iteration to resume')
flags.DEFINE_integer('continual_resume_shape', 14000, 'Second iteration to resume')
flags.DEFINE_integer('continual_resume_color', 30000, 'Second iteration to resume')

# Custom settings for cross product experiment
flags.DEFINE_float('ratio', 1.0, 'ratio of elements ot keep')

flags.DEFINE_bool('log_image', False, 'whether to log images of dual generation')

FLAGS = flags.FLAGS

# Hacky but this way I remember what models I used for each task
# if FLAGS.task == "infer_pos":
#     FLAGS.exp = "cubes_pos_813_longer"
#     FLAGS.resume_iter = 15000

# The swish activation is pretty much needed for these combination experiments
FLAGS.swish_act = True

def rescale_im(im):
    im = np.clip(im, 0, 1)
    return np.round(im * 255).astype(np.uint8)


def inference_position(dataloader, target_vars, sess):
    X, Y_pos = target_vars['X'], target_vars['Y_first']
    X_final = target_vars['X_final']

    # pos = np.array([[0.3, 0.3], [-0.3, 0.3], [-0.3, -0.3]])
    pos = np.array([[0.7, 0.7], [-0.7, 0.7], [-0.7, -0.7]])
    ims = sess.run([X_final], {X: np.random.uniform(0, 1, (3, 64, 64, 3)), Y_pos: pos})[0]
    print(ims.shape)

    imsave("inference_position.png", ims.reshape((192, 64, 3)))


def cross_train(dataloader, sess, target_vars):
    model = CubesPredict()
    weights = model.construct_weights('context_3')
    initialize()

    v_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='context_{}'.format(3))
    saver = tf.train.Saver(v_list)
    supervised_logdir = osp.join('cachedir', 'mujoco_supervised')
    model_file = osp.join(supervised_logdir, 'mujoco_supervised')

    X, label_pos, label_size = target_vars['X'], target_vars['Y_first'], target_vars['Y_second']

    logit_size, logit_pos = model.forward(X, weights)
    class_loss = tf.reduce_mean(tf.square(label_size - logit_size))
    l2_loss = tf.reduce_mean(tf.square(label_pos - logit_pos))
    loss = class_loss + l2_loss
    optimizer = tf.train.AdamOptimizer(1e-4)
    train_op = optimizer.minimize(loss)

    initialize()

    itr = 0

    for j in range(1):
        for _, im, label in tqdm(dataloader):
            # print(label[:5, -2:])
            # print(label[:5, :-2])
            _, loss_val = sess.run([train_op, loss], {X: im, label_pos:label[:, -2:], label_size:label[:, :-2]})

            if itr % 10 == 0:
                print("loss_val ", loss_val)

            itr += 1

    saver.save(sess, model_file)


def cross_benchmark(dataloader, target_vars, sess):
    model = CubesPredict()
    weights = model.construct_weights('context_3')

    initialize()
    supervised_logdir = osp.join('cachedir', 'mujoco_supervised')
    model_file = osp.join(supervised_logdir, 'mujoco_supervised')

    v_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='context_{}'.format(3))
    saver = tf.train.Saver(v_list)
    saver.restore(sess, model_file)

    X, Y_pos, Y_color = target_vars['X'], target_vars['Y_first'], target_vars['Y_second']
    x_final = target_vars['X_final']
    x_output = target_vars['x_output']
    NOISE = target_vars['NOISE']
    STEPS = target_vars['STEPS']

    logit_final, logit_pos_final = model.forward(x_final, weights)
    logit_output, logit_pos_output = model.forward(x_output, weights)

    logit_final, logit_output = logit_final, logit_output

    dual_mses = []
    gen_mses = []

    dual_pos_diff = []
    gen_pos_diff = []

    dual_prob = []
    gen_prob = []

    for data_corrupt, data, label  in tqdm(dataloader):
        data = data.numpy()
        label = label.numpy()
        pos_label = label[:, -2:]
        color_label = label[:, :-2]

        # print(color_label)
        # print(pos_label)
        # color_num = np.searchsorted(color_label, 1)
        color_num = np.argmax(color_label, axis=1)
        noise = np.random.uniform(-1.0, 1.0, (label.shape[0], 128))
        feed_dict = {Y_pos: pos_label, Y_color: color_label, X: data_corrupt, NOISE: noise, STEPS:FLAGS.num_steps}
        lf, lp, of, op, im_dual, im_gen = sess.run([logit_final, logit_pos_final, logit_output, logit_pos_output, x_final, x_output], feed_dict)

        if FLAGS.log_image:
            im_stack = np.stack([im_dual, im_gen, data], axis=1)
            im_stack = im_stack.transpose((0, 2, 1, 3, 4))
            im_stack = im_stack.reshape((FLAGS.batch_size * 64, 64 * 3, 3))
            imsave("im_final_{}.png".format(FLAGS.ratio), im_stack)

            assert False

        dual_pos_diff.append(np.abs(lp - pos_label).mean())
        gen_pos_diff.append(np.abs(op - pos_label).mean())

        dual_prob.append(np.abs(lf - color_label).mean())
        gen_prob.append(np.abs(of - color_label).mean())
        dual_mses.append(np.abs(im_dual - data).mean())
        gen_mses.append(np.abs(im_gen - data).mean())

        print("Average dual mse: ", np.mean(dual_mses))
        print("Average gen mse: ", np.mean(gen_mses))

        print("Average dual off: ", np.mean(dual_pos_diff))
        print("Average gen off: ", np.mean(gen_pos_diff))

        print("Average dual size: ", np.mean(dual_prob))
        print("Average gen size: ", np.mean(gen_prob))



def generalize_combine(dataloader, target_vars, sess, test_dataloader):
    X, Y_pos, Y_color = target_vars['X'], target_vars['Y_first'], target_vars['Y_second']
    X_final = target_vars['X_final']

    data_corrupt, data_color, color_label = iter(dataloader).next()
    data_corrupt, data_pos, pos_label = iter(test_dataloader).next()
    print("pos_label: ", pos_label[0])
    print("color_label: ", color_label[0])
    feed_dict = {X: data_corrupt, Y_pos: pos_label, Y_color: color_label}
    data_corrupt = sess.run([X_final], feed_dict)[0]
    imsave("generalize.png", np.concatenate([data_corrupt[0], data_color[0], data_pos[0]], axis=0))


def continual_combine(target_vars, sess):
    X, Y_pos, Y_shape = target_vars['X'], target_vars['Y_first'], target_vars['Y_second']
    Y_color = target_vars['Y_third']
    X_final = target_vars['X_final']

    # data_corrupt = np.random.uniform(0, 1, (20, 64, 64, 3)) / 5 + np.random.uniform(0, 1,  (20, 1, 1, 3)) * 4 / 5
    n = 64
    data_corrupt = np.random.uniform(0, 1, (n, 64, 64, 3))
    # pos_label = np.array([[-0.58974359, -0.58974359]])
    # shape_label = np.array([[0, 1]])
    # color_label = np.eye(20)[1:2]

    # bin_x = np.random.choice(np.linspace(-1, 1, 40)[-20:], size=(20,))
    # bin_y = np.random.choice(np.linspace(-1, 1, 40)[-20:], size=(20,))

    # pos_label = np.concatenate([bin_x[:, None], bin_y[:, None]], axis=1)
    pos_label = np.random.uniform(-1, 1, (n, 2))
    shape_label = np.tile(np.eye(2)[1:2], (n, 1))
    color_label = np.eye(20)[np.random.randint(2, 19, (n,))]

    print(pos_label)

    feed_dict = {X: data_corrupt, Y_pos: pos_label, Y_shape: shape_label, Y_color: color_label}
    data_corrupt = sess.run([X_final], feed_dict)[0]

    imsave("double_continual_large.png", data_corrupt.reshape(-1, 64, 3))

    assert False


def composition_figure(target_vars, sess):
    X, Y_pos, Y_shape = target_vars['X'], target_vars['Y_pos'], target_vars['Y_shape']
    Y_color = target_vars['Y_color']
    Y_size = target_vars['Y_size']
    X_final = target_vars['X_final']

    batch = 32
    data_corrupt = np.random.uniform(0, 1, (batch, 64, 64, 3)) / 2 + np.random.uniform(0, 1, (batch, 1, 1, 3)) * 1. / 2
    # pos_label = np.tile(np.array([[0.0, 0.0]]), (batch, 1))
    # shape_label = np.tile(np.eye(3)[1:2], (batch, 1))
    # color_label = np.tile(np.eye(20)[0:1], (batch, 1))
    # size_label = np.tile(np.array([[0.7]]), (batch, 1))

    pos_label = np.tile(np.array([[0.0, -0.8]]), (batch, 1))
    shape_label = np.tile(np.eye(3)[2:3], (batch, 1))
    color_label = np.tile(np.eye(20)[15:16], (batch, 1))
    size_label = np.tile(np.array([[0.4]]), (batch, 1))

    feed_dict = {X: data_corrupt, Y_pos: pos_label, Y_shape: shape_label, Y_color: color_label, Y_size: size_label}
    data_corrupt = sess.run([X_final], feed_dict)[0]

    data_corrupt = data_corrupt[:, ::-1]
    data_corrupt = data_corrupt.reshape((-1, 64, 3))

    imsave("2_composition_large.png", data_corrupt)

    assert False


def combine_construct(dataloader, weights, model, target_vars, logdir, sess):
    X, Y_first, Y_second = target_vars['X'], target_vars['Y_first'], target_vars['Y_second']
    # Y_third = target_vars['Y_third']
    X_final = target_vars['X_final']

    n = 16

    # For the first image
    # label_first = np.tile(np.array([[-0.7, -0.3, 0.3, 0.4, 0.4, 0.4]]), (n, 1))
    # label_second = np.tile(np.array([[0.75, 0.3, 0.4, 0.4, 0.4, 0.4]]), (n, 1))

    # For the second image
    # label_first = np.tile(np.array([[0.8, 0.8, 0.45, 0.45, 0.45, 0.45]]), (n, 1))
    # label_second = np.tile(np.array([[0.8, -0.8, 0.45, 0.45, 0.45, 0.45]]), (n, 1))

    # For the third image
    # label_first = np.tile(np.array([[-0.8, 0.8, 0.25, 0.25, 0.25, 0.25]]), (n, 1))
    # label_second = np.tile(np.array([[0.8, -0.8, 0.5, 0.5, 0.5, 0.5]]), (n, 1))

    # For the final image
    label_first = np.tile(np.array([[-0.1, -0.1, 0.4, 0.4, 0.4, 0.4]]), (n, 1))
    label_second = np.tile(np.array([[0.1, 0.1, 0.4, 0.4, 0.4, 0.4]]), (n, 1))

    data_corrupt = np.random.uniform(0, 1, (n, 64, 64, 3)) / 3. + np.random.uniform(0, 1, (n, 1, 1, 3)) * 2 / 3.

    feed_dict = {X: data_corrupt, Y_first: label_first, Y_second: label_second}
    data_corrupt = sess.run([X_final], feed_dict)[0]

    # imsave("combine.png", data_corrupt[0])

    for i in range(n):
        imsave("single_{}.png".format(i), data_corrupt[i])

    assert False


def construct_steps(weights, model, target_vars):
    steps = tf.constant(0)
    c = lambda i, x: tf.less(i, FLAGS.num_steps)
    X, Y_first, Y_second = target_vars['X'], target_vars['Y_first'], target_vars['Y_second']
    # Y_third = target_vars['Y_third']
    attention_mask = tf.zeros(1)

    def langevin_step(counter, x_mod):
        x_mod = x_mod + tf.random_normal(tf.shape(x_mod),
                                         mean=0.0,
                                         stddev=0.001)
        # latent = latent + tf.random_normal(tf.shape(latent), mean=0.0, stddev=0.01)
        energy_noise = model.forward(
                    x_mod,
                    weights,
                    label=Y_first,
                    reuse=True,
                    stop_at_grad=False,
                    stop_batch=True,
                    attention_mask=attention_mask)
        # energy_noise = tf.Print(energy_noise, [energy_noise, x_mod, latent])
        x_grad = tf.gradients(energy_noise, [x_mod])[0]

        x_mod = x_mod - FLAGS.step_lr * x_grad
        x_mod = tf.clip_by_value(x_mod, 0, 1)

        #####################3
        x_mod = x_mod + tf.random_normal(tf.shape(x_mod), mean=0.0, stddev=0.001)

        energy_noise = model.forward(
                    x_mod,
                    weights,
                    label=Y_second,
                    reuse=True,
                    stop_at_grad=False,
                    stop_batch=True,
                    attention_mask=attention_mask)

        x_grad = tf.gradients(energy_noise, [x_mod])[0]

        x_mod = x_mod - FLAGS.step_lr * x_grad
        x_mod = tf.clip_by_value(x_mod, 0, 1)

        x_mod = x_mod + tf.random_normal(tf.shape(x_mod), mean=0.0, stddev=0.001)

        # energy_noise = model.forward(
        #             x_mod,
        #             weights,
        #             label=Y_third,
        #             reuse=True,
        #             stop_at_grad=False,
        #             stop_batch=True,
        #             attention_mask=attention_mask)

        # x_grad = tf.gradients(energy_noise, [x_mod])[0]

        # x_mod = x_mod - FLAGS.step_lr * x_grad
        # x_mod = tf.clip_by_value(x_mod, 0, 1)

        counter = counter + 1

        return counter, x_mod

    steps, x_mod = tf.while_loop(c, langevin_step, (steps, X))

    target_vars['X_final'] = x_mod


def construct_step(weights, model, target_vars):
    steps = tf.constant(0)
    c = lambda i, x: tf.less(i, FLAGS.num_steps)
    X, Y_first = target_vars['X'], target_vars['Y_first']
    # Y_third = target_vars['Y_third']
    attention_mask = tf.zeros(1)

    def langevin_step(counter, x_mod):
        x_mod = x_mod + tf.random_normal(tf.shape(x_mod),
                                         mean=0.0,
                                         stddev=0.001)
        # latent = latent + tf.random_normal(tf.shape(latent), mean=0.0, stddev=0.01)
        energy_noise = model.forward(
                    x_mod,
                    weights,
                    label=Y_first,
                    reuse=True,
                    stop_at_grad=False,
                    stop_batch=True,
                    attention_mask=attention_mask)
        x_grad = tf.gradients(energy_noise, [x_mod])[0]

        x_mod = x_mod - FLAGS.step_lr * x_grad
        x_mod = tf.clip_by_value(x_mod, 0, 1)

        #####################3
        # x_mod = x_mod + tf.random_normal(tf.shape(x_mod), mean=0.0, stddev=0.001)

        # energy_noise = model.forward(
        #             x_mod,
        #             weights,
        #             label=Y_second,
        #             reuse=True,
        #             stop_at_grad=False,
        #             stop_batch=True,
        #             attention_mask=attention_mask)

        # x_grad = tf.gradients(energy_noise, [x_mod])[0]

        # x_mod = x_mod - FLAGS.step_lr * x_grad
        # x_mod = tf.clip_by_value(x_mod, 0, 1)

        # x_mod = x_mod + tf.random_normal(tf.shape(x_mod), mean=0.0, stddev=0.001)

        # energy_noise = model.forward(
        #             x_mod,
        #             weights,
        #             label=Y_third,
        #             reuse=True,
        #             stop_at_grad=False,
        #             stop_batch=True,
        #             attention_mask=attention_mask)

        # x_grad = tf.gradients(energy_noise, [x_mod])[0]

        # x_mod = x_mod - FLAGS.step_lr * x_grad
        # x_mod = tf.clip_by_value(x_mod, 0, 1)

        counter = counter + 1

        return counter, x_mod

    steps, x_mod = tf.while_loop(c, langevin_step, (steps, X))

    target_vars['X_final'] = x_mod

def construct_energy(weights, model, target_vars):
    X, Y_pos = target_vars['X'], target_vars['Y_first']
    energy = model.forward(
                X,
                weights,
                label=Y_pos,
                reuse=True,
                attention_mask=tf.zeros(1)
                )
    target_vars['energy'] = energy


def construct_steps_gen(weights_gen, model_gen, target_vars):
    Y_first, Y_second = target_vars['Y_first'], target_vars['Y_second']
    NOISE = target_vars['NOISE']
    LABEL = tf.concat([Y_second, Y_first], axis=1)
    x_gen_output = model_gen.forward(NOISE, weights_gen, reuse=False, label=LABEL)
    target_vars['x_output'] = x_gen_output


def finetune_model(dataloader, target_vars, sess):
    n = 2
    X, Y_pos, Y_color = target_vars['X'], target_vars['Y_first'], target_vars['Y_second']
    X_feed = target_vars['X_feed']
    train_op, energy_neg, energy_pos = target_vars['train_op'], target_vars['energy_neg'], target_vars['energy_pos']
    x_off = target_vars['x_off']
    STEPS = target_vars['STEPS']
    log_output = [energy_neg, energy_pos, x_off, train_op]

    counter = 0
    for i in range(n):
        for data_corrupt, data_pos, label in tqdm(dataloader):
            data_corrupt, data_pos, label = data_corrupt.numpy(), data_pos.numpy(), label.numpy()
            pos_label = label[:, -2:]
            color_label = label[:, :-2]

            if counter % 10 == 0:
                energy_neg, energy_pos, x_off, _  = sess.run(log_output, {X: data_corrupt, Y_pos: pos_label, Y_color: color_label, X_feed: data_pos, STEPS: 200})
                print("Energy neg {}, energy pos {} x_off {}".format(energy_neg, energy_pos, x_off))
            else:
                _ = sess.run([train_op], {X: data_pos, Y_pos: pos_label, Y_color: color_label, X_feed: data_pos, STEPS: 200})[0]

            # cunter = 800 for ratio 0.5
            if counter > 90:
                break
            counter += 1


def construct_steps_dual(weights_pos, weights_color, model_pos, model_color, target_vars):
    steps = tf.constant(0)
    STEPS = target_vars['STEPS']
    c = lambda i, x: tf.less(i, STEPS)
    X, Y_first, Y_second = target_vars['X'], target_vars['Y_first'], target_vars['Y_second']
    X_feed = target_vars['X_feed']
    attention_mask = tf.zeros(1)

    def langevin_step(counter, x_mod):
        x_mod = x_mod + tf.random_normal(tf.shape(x_mod),
                                         mean=0.0,
                                         stddev=0.001)
        # latent = latent + tf.random_normal(tf.shape(latent), mean=0.0, stddev=0.01)
        energy_noise = model_pos.forward(
                    x_mod,
                    weights_pos,
                    label=Y_first,
                    reuse=True,
                    stop_at_grad=False,
                    stop_batch=True,
                    attention_mask=attention_mask)
        # energy_noise = tf.Print(energy_noise, [energy_noise, x_mod, latent])
        x_grad = tf.gradients(energy_noise, [x_mod])[0]

        x_mod = x_mod - FLAGS.step_lr * x_grad
        x_mod = tf.clip_by_value(x_mod, 0, 1)

        x_mod = x_mod + tf.random_normal(tf.shape(x_mod), mean=0.0, stddev=0.001)

        energy_noise = 1.0 * model_color.forward(
                    x_mod,
                    weights_color,
                    label=Y_second,
                    reuse=True,
                    stop_at_grad=False,
                    stop_batch=True,
                    attention_mask=attention_mask)

        x_grad = tf.gradients(energy_noise, [x_mod])[0]

        x_mod = x_mod - FLAGS.step_lr * x_grad
        x_mod = tf.clip_by_value(x_mod, 0, 1)

        counter = counter + 1
        # counter = tf.Print(counter, [counter], message="step")

        return counter, x_mod

    def langevin_merge_step(counter, x_mod):
        x_mod = x_mod + tf.random_normal(tf.shape(x_mod),
                                         mean=0.0,
                                         stddev=0.001)
        # latent = latent + tf.random_normal(tf.shape(latent), mean=0.0, stddev=0.01)
        energy_noise = 1 * model_pos.forward(
                    x_mod,
                    weights_pos,
                    label=Y_first,
                    reuse=True,
                    stop_at_grad=False,
                    stop_batch=True,
                    attention_mask=attention_mask) + \
                    1.0 * model_color.forward(
                    x_mod,
                    weights_color,
                    label=Y_second,
                    reuse=True,
                    stop_at_grad=False,
                    stop_batch=True,
                    attention_mask=attention_mask)

        # energy_noise = tf.Print(energy_noise, [energy_noise, x_mod, latent])
        x_grad = tf.gradients(energy_noise, [x_mod])[0]

        x_mod = x_mod - FLAGS.step_lr * x_grad
        x_mod = tf.clip_by_value(x_mod, 0, 1)

        counter = counter + 1

        return counter, x_mod

    steps, x_mod = tf.while_loop(c, langevin_merge_step, (steps, X))

    energy_pos = model_pos.forward(
                tf.stop_gradient(x_mod),
                weights_pos,
                label=Y_first,
                reuse=True,
                stop_at_grad=False,
                stop_batch=True,
                attention_mask=attention_mask)

    energy_color = model_color.forward(
                tf.stop_gradient(x_mod),
                weights_color,
                label=Y_second,
                reuse=True,
                stop_at_grad=False,
                stop_batch=True,
                attention_mask=attention_mask)

    energy_plus_pos = model_pos.forward(
                X_feed,
                weights_pos,
                label=Y_first,
                reuse=True,
                stop_at_grad=False,
                stop_batch=True,
                attention_mask=attention_mask)

    energy_plus_color = model_color.forward(
                X_feed,
                weights_color,
                label=Y_second,
                reuse=True,
                stop_at_grad=False,
                stop_batch=True,
                attention_mask=attention_mask)

    energy_neg = -tf.reduce_mean(tf.reduce_mean(energy_pos) + tf.reduce_mean(energy_color))
    energy_plus = tf.reduce_mean(tf.reduce_mean(energy_plus_pos) + tf.reduce_mean(energy_plus_color))
    loss_l2 = tf.reduce_mean(tf.square(energy_pos)) + tf.reduce_mean(tf.square(energy_color)) + tf.reduce_mean(tf.square(energy_plus_pos)) + tf.reduce_mean(tf.square(energy_plus_color))

    loss_total = energy_plus + energy_neg + loss_l2
    optimizer = AdamOptimizer(1e-4, beta1=0.0, beta2=0.99)
    gvs = optimizer.compute_gradients(loss_total)
    train_op = optimizer.apply_gradients(gvs)

    x_off = tf.reduce_mean(tf.abs(X_feed - x_mod))

    target_vars['X_final'] = x_mod
    target_vars['x_off'] = x_off
    target_vars['train_op'] = train_op
    target_vars['energy_neg'] = -energy_neg
    target_vars['energy_pos'] = energy_plus


def construct_steps_triple(weights_pos, weights_shape, weights_color, model_pos, model_shape, model_color, target_vars):
    steps = tf.constant(0)
    c = lambda i, x: tf.less(i, FLAGS.num_steps)
    X, Y_pos, Y_shape, Y_color = target_vars['X'], target_vars['Y_first'], target_vars['Y_second'], target_vars['Y_third']
    attention_mask = tf.zeros(1)

    def langevin_step(counter, x_mod):
        x_mod = x_mod + tf.random_normal(tf.shape(x_mod),
                                         mean=0.0,
                                         stddev=0.001)
        energy_noise = model_pos.forward(
                    x_mod,
                    weights_pos,
                    label=Y_pos,
                    reuse=True,
                    stop_at_grad=False,
                    stop_batch=True,
                    attention_mask=attention_mask)

        # energy_noise = model_shape.forward(
        #             x_mod,
        #             weights_shape,
        #             label=Y_shape,
        #             reuse=True,
        #             stop_at_grad=False,
        #             stop_batch=True,
        #             attention_mask=attention_mask) + energy_noise

        # energy_noise = model_color.forward(
        #             x_mod,
        #             weights_color,
        #             label=Y_color,
        #             reuse=True,
        #             stop_at_grad=False,
        #             stop_batch=True,
        #             attention_mask=attention_mask) + energy_noise

        # energy_noise = tf.Print(energy_noise, [energy_noise, x_mod, latent])
        x_grad = tf.gradients(energy_noise, [x_mod])[0]

        x_mod = x_mod - FLAGS.step_lr * x_grad
        x_mod = tf.clip_by_value(x_mod, 0, 1)

        x_mod = x_mod + tf.random_normal(tf.shape(x_mod), mean=0.0, stddev=0.001)

        energy_noise = model_shape.forward(
                    x_mod,
                    weights_shape,
                    label=Y_shape,
                    reuse=True,
                    stop_at_grad=False,
                    stop_batch=True,
                    attention_mask=attention_mask)

        x_grad = tf.gradients(energy_noise, [x_mod])[0]

        x_mod = x_mod - FLAGS.step_lr * x_grad
        x_mod = tf.clip_by_value(x_mod, 0, 1)

        # x_mod = x_mod + tf.random_normal(tf.shape(x_mod), mean=0.0, stddev=0.001)

        # energy_noise = model_color.forward(
        #             x_mod,
        #             weights_color,
        #             label=Y_color,
        #             reuse=True,
        #             stop_at_grad=False,
        #             stop_batch=True,
        #             attention_mask=attention_mask)

        # x_grad = tf.gradients(energy_noise, [x_mod])[0]

        # x_mod = x_mod - FLAGS.step_lr * x_grad
        # x_mod = tf.clip_by_value(x_mod, 0, 1)

        counter = counter + 1
        # counter = tf.Print(counter, [counter], message="step")

        return counter, x_mod

    steps, x_mod = tf.while_loop(c, langevin_step, (steps, X))

    target_vars['X_final'] = x_mod


def construct_steps_composition(weights_pos, weights_shape, weights_color, weights_size, model_pos, model_shape, model_color, model_size, target_vars):
    steps = tf.constant(0)
    c = lambda i, x: tf.less(i, FLAGS.num_steps)
    X, Y_pos, Y_shape, Y_color, Y_size = target_vars['X'], target_vars['Y_pos'], target_vars['Y_shape'], target_vars['Y_color'], target_vars['Y_size']
    attention_mask = tf.zeros(1)

    def langevin_step(counter, x_mod):
        #############################
        x_mod = x_mod + tf.random_normal(tf.shape(x_mod), mean=0.0, stddev=0.001)

        energy_noise = model_shape.forward(
                    x_mod,
                    weights_shape,
                    label=Y_shape,
                    reuse=True,
                    stop_at_grad=False,
                    stop_batch=True,
                    attention_mask=attention_mask)

        energy_noise = model_pos.forward(
                    x_mod,
                    weights_pos,
                    label=Y_pos,
                    reuse=True,
                    stop_at_grad=False,
                    stop_batch=True,
                    attention_mask=attention_mask) + energy_noise

        # energy_noise = model_size.forward(
        #             x_mod,
        #             weights_size,
        #             label=Y_size,
        #             reuse=True,
        #             stop_at_grad=False,
        #             stop_batch=True,
        #             attention_mask=attention_mask) + energy_noise

        # energy_noise = model_color.forward(
        #             x_mod,
        #             weights_color,
        #             label=Y_color,
        #             reuse=True,
        #             stop_at_grad=False,
        #             stop_batch=True,
        #             attention_mask=attention_mask) + energy_noise

        x_grad = tf.gradients(energy_noise, [x_mod])[0]

        x_mod = x_mod - FLAGS.step_lr * x_grad
        x_mod = tf.clip_by_value(x_mod, 0, 1)

        #############################
        # x_mod = x_mod + tf.random_normal(tf.shape(x_mod),
        #                                  mean=0.0,
        #                                  stddev=0.001)
        # energy_noise = model_pos.forward(
        #             x_mod,
        #             weights_pos,
        #             label=Y_pos,
        #             reuse=True,
        #             stop_at_grad=False,
        #             stop_batch=True,
        #             attention_mask=attention_mask)
        # x_grad = tf.gradients(energy_noise, [x_mod])[0]

        # x_mod = x_mod - FLAGS.step_lr * x_grad
        # x_mod = tf.clip_by_value(x_mod, 0, 1)

        # #############################
        # x_mod = x_mod + tf.random_normal(tf.shape(x_mod), mean=0.0, stddev=0.001)

        # energy_noise = model_size.forward(
        #             x_mod,
        #             weights_size,
        #             label=Y_size,
        #             reuse=True,
        #             stop_at_grad=False,
        #             stop_batch=True,
        #             attention_mask=attention_mask)

        # x_grad = tf.gradients(energy_noise, [x_mod])[0]

        # x_mod = x_mod - FLAGS.step_lr * x_grad
        # x_mod = tf.clip_by_value(x_mod, 0, 1)

        # #############################
        # x_mod = x_mod + tf.random_normal(tf.shape(x_mod), mean=0.0, stddev=0.001)

        # energy_noise = model_color.forward(
        #             x_mod,
        #             weights_color,
        #             label=Y_color,
        #             reuse=True,
        #             stop_at_grad=False,
        #             stop_batch=True,
        #             attention_mask=attention_mask)

        # x_grad = tf.gradients(energy_noise, [x_mod])[0]

        # x_mod = x_mod - FLAGS.step_lr * x_grad
        # x_mod = tf.clip_by_value(x_mod, 0, 1)

        counter = counter + 1
        # counter = tf.Print(counter, [counter], message="step")

        return counter, x_mod

    steps, x_mod = tf.while_loop(c, langevin_step, (steps, X))

    target_vars['X_final'] = x_mod

def main():

    if FLAGS.task == "concept_combine":
        dataset = CubesColor()
        test_dataset = CubesPos()
    elif FLAGS.task == "cross_benchmark":
        dataset = CubesCrossProduct(FLAGS.ratio, joint_baseline=True)
        test_dataset = CubesCrossProduct(FLAGS.ratio, joint_baseline=True, inversion=True)
        # test_dataset = CubesCrossProduct(1, joint_baseline=True)
    else:
        dataset = Cubes(cond_idx=0)
        test_dataset = dataset

    dataloader = DataLoader(dataset, batch_size=FLAGS.batch_size, num_workers=FLAGS.data_workers, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=FLAGS.batch_size, num_workers=FLAGS.data_workers, shuffle=True, drop_last=True)

    if FLAGS.task == "infer_pos":
        dataset = CubesPos()
        dataloader = DataLoader(dataset, batch_size=1, num_workers=FLAGS.data_workers, shuffle=True, drop_last=True)

    hidden_dim = 128

    if FLAGS.task == "concept_combine":
        model_pos = CubesNet(label_size=2)
        model_color = CubesNet(label_size=301)
        weights_pos = model_pos.construct_weights('context_{}'.format(0))
        weights_color = model_color.construct_weights('context_{}'.format(1))

        config = tf.ConfigProto()
        sess = tf.InteractiveSession()

        X = tf.placeholder(shape=(None, 64, 64, 3), dtype = tf.float32)
        Y_first = tf.placeholder(shape=(None, 2), dtype = tf.float32)
        Y_second = tf.placeholder(shape=(None, 301), dtype = tf.float32)

        target_vars = {'X': X, 'Y_first': Y_first, 'Y_second': Y_second}
        construct_steps_dual(weights_pos, weights_color, model_pos, model_color, target_vars)

        initialize()
        save_path_color = osp.join(FLAGS.logdir, FLAGS.exp_color, 'model_{}'.format(FLAGS.resume_color))
        save_path_pos = osp.join(FLAGS.logdir, FLAGS.exp_pos, 'model_{}'.format(FLAGS.resume_pos))
        v_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='context_{}'.format(0))
        v_map = {(v.name.replace('context_{}'.format(0), 'context_0')[:-2]): v for v in v_list}
        saver = tf.train.Saver(v_map)
        saver.restore(sess, save_path_pos)

        v_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='context_{}'.format(1))
        v_map = {(v.name.replace('context_{}'.format(1), 'context_0')[:-2]): v for v in v_list}
        saver = tf.train.Saver(v_map)
        print("path color ", save_path_color)
        saver.restore(sess, save_path_color)

    elif FLAGS.task == "continual":
        model_pos = CubesNet(label_size=2)
        model_shape = CubesNet(label_size=2)
        model_color = CubesNet(label_size=20)
        weights_pos = model_pos.construct_weights('context_{}'.format(0))
        weights_shape = model_shape.construct_weights('context_{}'.format(1))
        weights_color = model_color.construct_weights('context_{}'.format(2))

        config = tf.ConfigProto()
        sess = tf.InteractiveSession()

        X = tf.placeholder(shape=(None, 64, 64, 3), dtype = tf.float32)
        Y_first = tf.placeholder(shape=(None, 2), dtype = tf.float32)
        Y_second = tf.placeholder(shape=(None, 2), dtype = tf.float32)
        Y_third = tf.placeholder(shape=(None, 20), dtype = tf.float32)

        target_vars = {'X': X, 'Y_first': Y_first, 'Y_second': Y_second, 'Y_third': Y_third}
        construct_steps_triple(weights_pos, weights_shape, weights_color, model_pos, model_shape,  model_color, target_vars)

        initialize()
        save_path_shape = osp.join(FLAGS.logdir, FLAGS.continual_shape, 'model_{}'.format(FLAGS.continual_resume_shape))
        save_path_pos = osp.join(FLAGS.logdir, FLAGS.continual_pos, 'model_{}'.format(FLAGS.continual_resume_pos))
        save_path_color = osp.join(FLAGS.logdir, FLAGS.continual_color, 'model_{}'.format(FLAGS.continual_resume_color))
        v_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='context_{}'.format(0))
        v_map = {(v.name.replace('context_{}'.format(0), 'context_0')[:-2]): v for v in v_list}
        saver = tf.train.Saver(v_map)
        saver.restore(sess, save_path_pos)

        v_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='context_{}'.format(1))
        v_map = {(v.name.replace('context_{}'.format(1), 'context_0')[:-2]): v for v in v_list}
        saver = tf.train.Saver(v_map)
        saver.restore(sess, save_path_shape)

        v_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='context_{}'.format(2))
        v_map = {(v.name.replace('context_{}'.format(2), 'context_0')[:-2]): v for v in v_list}
        saver = tf.train.Saver(v_map)
        saver.restore(sess, save_path_color)


    elif FLAGS.task == "composition_figure":
        model_pos = CubesNet(label_size=2)
        model_shape = CubesNet(label_size=3)
        model_color = CubesNet(label_size=20)
        model_size = CubesNet(label_size=1)
        weights_pos = model_pos.construct_weights('context_{}'.format(0))
        weights_shape = model_shape.construct_weights('context_{}'.format(1))
        weights_color = model_color.construct_weights('context_{}'.format(2))
        weights_size = model_size.construct_weights('context_{}'.format(3))

        config = tf.ConfigProto()
        sess = tf.InteractiveSession()

        X = tf.placeholder(shape=(None, 64, 64, 3), dtype = tf.float32)
        Y_pos = tf.placeholder(shape=(None, 2), dtype = tf.float32)
        Y_shape = tf.placeholder(shape=(None, 3), dtype = tf.float32)
        Y_color = tf.placeholder(shape=(None, 20), dtype = tf.float32)
        Y_size = tf.placeholder(shape=(None, 1), dtype = tf.float32)

        target_vars = {'X': X, 'Y_pos': Y_pos, 'Y_shape': Y_shape, 'Y_color': Y_color, 'Y_size': Y_size}
        construct_steps_composition(weights_pos, weights_shape, weights_color, weights_size, model_pos, model_shape,  model_color, model_size, target_vars)

        initialize()
        save_path_shape = osp.join(FLAGS.logdir, '913_cube_general_type', 'model_{}'.format(30000))
        save_path_pos = osp.join(FLAGS.logdir, '913_cube_general_pos', 'model_{}'.format(30000))
        save_path_color = osp.join(FLAGS.logdir, '913_cube_general_color', 'model_{}'.format(30000))
        save_path_size = osp.join(FLAGS.logdir, '913_cube_general_size', 'model_{}'.format(30000))
        v_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='context_{}'.format(0))
        v_map = {(v.name.replace('context_{}'.format(0), 'context_0')[:-2]): v for v in v_list}
        saver = tf.train.Saver(v_map)
        saver.restore(sess, save_path_pos)

        v_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='context_{}'.format(1))
        v_map = {(v.name.replace('context_{}'.format(1), 'context_0')[:-2]): v for v in v_list}
        saver = tf.train.Saver(v_map)
        saver.restore(sess, save_path_shape)

        v_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='context_{}'.format(2))
        v_map = {(v.name.replace('context_{}'.format(2), 'context_0')[:-2]): v for v in v_list}
        saver = tf.train.Saver(v_map)
        saver.restore(sess, save_path_color)

        v_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='context_{}'.format(3))
        v_map = {(v.name.replace('context_{}'.format(3), 'context_0')[:-2]): v for v in v_list}
        saver = tf.train.Saver(v_map)
        saver.restore(sess, save_path_size)

    elif FLAGS.task == "multi":
        model = CubesNet(num_channels=3)
        weights = model.construct_weights('context_{}'.format(0))

        config = tf.ConfigProto()
        sess = tf.InteractiveSession()

        X = tf.placeholder(shape=(None, 64, 64, 3), dtype = tf.float32)
        Y_first = tf.placeholder(shape=(None, 6), dtype = tf.float32)
        Y_second = tf.placeholder(shape=(None, 6), dtype = tf.float32)

        target_vars = {'X': X, 'Y_first': Y_first, 'Y_second': Y_second}

        construct_steps(weights, model, target_vars)
        initialize()

    elif FLAGS.task == "cross_benchmark":
        model_pos = CubesNet(label_size=2)
        model_color = CubesNet(label_size=1)
        model_gen = CubesNetGen(label_size=3)

        weights_pos = model_pos.construct_weights('context_{}'.format(0))
        weights_color = model_color.construct_weights('context_{}'.format(1))
        weights_gen = model_gen.construct_weights('context_{}'.format(2))

        config = tf.ConfigProto()
        sess = tf.InteractiveSession()

        X = tf.placeholder(shape=(None, 64, 64, 3), dtype = tf.float32)
        X_feed = tf.placeholder(shape=(None, 64, 64, 3), dtype = tf.float32)
        Y_first = tf.placeholder(shape=(None, 2), dtype = tf.float32)
        Y_second = tf.placeholder(shape=(None, 1), dtype = tf.float32)
        STEPS = tf.placeholder(shape=None, dtype = tf.int32)
        NOISE = tf.placeholder(shape=(None, 128), dtype = tf.float32)

        target_vars = {'X': X, 'Y_first': Y_first, 'Y_second': Y_second, 'NOISE': NOISE, 'X_feed': X_feed, 'STEPS': STEPS}

        construct_steps_dual(weights_pos, weights_color, model_pos, model_color, target_vars)
        construct_steps_gen(weights_gen, model_gen, target_vars)
        initialize()

        if FLAGS.ratio == 0.001:
            ratio_string = "0001"
        if FLAGS.ratio == 0.01:
            ratio_string = "001"
        if FLAGS.ratio == 0.03:
            ratio_string = "003"
        elif FLAGS.ratio == 0.05:
            ratio_string = "005"
        elif FLAGS.ratio == 0.1:
            ratio_string = "01"
        elif FLAGS.ratio == 0.3:
            ratio_string = "03"
        elif FLAGS.ratio == 0.5:
            ratio_string = "05"
        elif FLAGS.ratio == 1:
            ratio_string = "1"

        resume_iter = 30000

        save_path_color = osp.join(FLAGS.logdir, 'cross_917_{}_cond_size'.format(ratio_string), 'model_{}'.format(resume_iter))
        save_path_pos = osp.join(FLAGS.logdir, 'cross_917_{}_cond_pos'.format(ratio_string), 'model_{}'.format(resume_iter))
        save_path_gen = osp.join(FLAGS.logdir, 'cross_917_{}_joint_baseline'.format(ratio_string), 'model_{}'.format(30000))

        v_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='context_{}'.format(0))
        v_map = {(v.name.replace('context_{}'.format(0), 'context_0')[:-2]): v for v in v_list}
        saver = tf.train.Saver(v_map)
        saver.restore(sess, save_path_pos)

        v_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='context_{}'.format(1))
        v_map = {(v.name.replace('context_{}'.format(1), 'context_0')[:-2]): v for v in v_list}
        saver = tf.train.Saver(v_map)
        saver.restore(sess, save_path_color)

        v_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='context_{}'.format(2))
        v_map = {(v.name.replace('context_{}'.format(2), 'context_0')[:-2]): v for v in v_list}
        saver = tf.train.Saver(v_map)
        saver.restore(sess, save_path_gen)

    elif FLAGS.task == "infer_pos":
        model = CubesNet(label_size=2)
        weights = model.construct_weights('context_{}'.format(0))

        config = tf.ConfigProto()
        sess = tf.InteractiveSession()

        X = tf.placeholder(shape=(None, 64, 64, 3), dtype = tf.float32)
        Y_first = tf.placeholder(shape=(None, 2), dtype = tf.float32)

        target_vars = {'X': X, 'Y_first': Y_first}
        construct_energy(weights, model, target_vars)
        construct_step(weights, model, target_vars)
        initialize()

    saver = loader = tf.train.Saver(max_to_keep=10)
    savedir = osp.join('cachedir', FLAGS.exp)
    logdir = osp.join(FLAGS.logdir, FLAGS.exp)
    if not osp.exists(logdir):
        os.makedirs(logdir)

    if FLAGS.resume_iter != -1:
        model_file = osp.join(savedir, 'model_{}'.format(FLAGS.resume_iter))
        resume_itr = FLAGS.resume_iter
        saver.restore(sess, model_file)

    print("This is our task ", FLAGS.task)
    if FLAGS.task == "multi":
        combine_construct(test_dataloader, weights, model, target_vars, logdir, sess)
    elif FLAGS.task == "concept_combine":
        generalize_combine(dataloader, target_vars, sess, test_dataloader)
    elif FLAGS.task == "continual":
        continual_combine(target_vars, sess)
    elif FLAGS.task == "infer_pos":
        inference_position(dataloader, target_vars, sess)
    elif FLAGS.task == "cross_benchmark":
        # cross_train(test_dataloader, sess, target_vars)
        finetune_model(dataloader, target_vars, sess)
        cross_benchmark(test_dataloader, target_vars, sess)
    elif FLAGS.task == "composition_figure":
        composition_figure(target_vars, sess)


if __name__ == "__main__":
    main()
