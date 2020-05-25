from tensorflow.python.platform import flags
from tensorflow.contrib.data.python.ops import batching, threadpool
import tensorflow as tf
import json
from torch.utils.data import Dataset
import pickle
import os.path as osp
import os
import numpy as np
import time
from scipy.misc import imread, imresize
from skimage.color import rgb2grey
from torchvision.datasets import CIFAR10, MNIST, SVHN, CIFAR100, ImageFolder
from torchvision import transforms
import torch
import torchvision
from itertools import product
import random

FLAGS = flags.FLAGS

# Dataset Options
flags.DEFINE_string('dsprites_path',
    '/root/data/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz',
    'path to dsprites characters')
flags.DEFINE_string('imagenet_datadir',  '/root/imagenet_big', 'whether cutoff should always in image')
flags.DEFINE_bool('dshape_only', False, 'fix all factors except for shapes')
flags.DEFINE_bool('dpos_only', False, 'fix all factors except for positions of shapes')
flags.DEFINE_bool('dsize_only', False,'fix all factors except for size of objects')
flags.DEFINE_bool('drot_only', False, 'fix all factors except for rotation of objects')
flags.DEFINE_bool('dsprites_restrict', False, 'fix all factors except for rotation of objects')
flags.DEFINE_string('imagenet_path', '/root/imagenet', 'path to imagenet images')
flags.DEFINE_string('cubes_path', 'cubes_varied_junk_801_different.npz', 'path to cube dataset')


# Data augmentation options
flags.DEFINE_bool('cutout_inside', False,'whether cutoff should always in image')
flags.DEFINE_float('cutout_prob', 1.0, 'probability of using cutout')
flags.DEFINE_integer('cutout_mask_size', 16, 'size of cutout')
flags.DEFINE_bool('cutout', False,'whether to add cutout regularizer to data')

# Custom pair hyperparameters
flags.DEFINE_integer('pair_cond_shape', -1, 'Only use a particular shape to train energy based models (-1 uses all shape)')


def cutout(mask_color=(0, 0, 0)):
    mask_size_half = FLAGS.cutout_mask_size // 2
    offset = 1 if FLAGS.cutout_mask_size % 2 == 0 else 0

    def _cutout(image):
        image = np.asarray(image).copy()

        if np.random.random() > FLAGS.cutout_prob:
            return image

        h, w = image.shape[:2]

        if FLAGS.cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + FLAGS.cutout_mask_size
        ymax = ymin + FLAGS.cutout_mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[:, ymin:ymax, xmin:xmax] = np.array(mask_color)[:, None, None]
        return image

    return _cutout



class Cubes(Dataset):
    def __init__(self, cond_idx=-1):
        dat = np.load("cubes_general.npz")
        self.data = dat['ims']
        self.label = dat['labels']
        self.cond_idx = cond_idx

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        cond_idx = self.cond_idx
        im = self.data[index] / 255.

        if self.cond_idx == 0:
            # If 0 then is position
            label = self.label[index, :2]
        elif self.cond_idx == 1:
            # If 1 then is size
            label = self.label[index, 2:3]
        elif self.cond_idx == 2:
            # If 2 then is shape
            label = np.eye(3)[self.label[index, 3].astype(np.int32)]
        elif self.cond_idx == 3:
            # if 3 then is color
            label = np.eye(20)[self.label[index, 4].astype(np.int32)]

        image_size = 64

        if FLAGS.datasource == 'default':
            im_corrupt = im + 0.3 * np.random.randn(image_size, image_size, 3)
        elif FLAGS.datasource == 'random':
            im_corrupt = 0.5 + 0.5 * np.random.randn(image_size, image_size, 3)

        return im_corrupt, im, label


class CubesPos(Dataset):
    def __init__(self):
        dat = np.load("cubes_position.npz")
        self.data = dat['ims']
        self.label = dat['labels']

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        im = self.data[index] / 255.
        label = self.label[index]

        image_size = 64

        if FLAGS.datasource == 'default':
            im_corrupt = im + 0.3 * np.random.randn(image_size, image_size, 3)
        elif FLAGS.datasource == 'random':
            im_corrupt = 0.5 + 0.5 * np.random.randn(image_size, image_size, 3)

        return im_corrupt, im, label


class CubesColor(Dataset):
    def __init__(self):
        dat = np.load("cubes_color.npz")
        self.data = dat['ims']
        self.label = dat['labels']
        self.eye = np.eye(301)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        im = self.data[index] / 255.

        label = self.eye[int(self.label[index])]
        image_size = 64

        if FLAGS.datasource == 'default':
            im_corrupt = im + 0.3 * np.random.randn(image_size, image_size, 3)
        elif FLAGS.datasource == 'random':
            im_corrupt = 0.5 + 0.5 * np.random.randn(image_size, image_size, 3)

        return im_corrupt, im, label


class CubesContinual(Dataset):
    def __init__(self):
        dat = np.load("cubes_continual.npz")
        self.data = dat['ims']
        self.label = dat['labels']
        self.color_eye = np.eye(20)
        self.shape_eye = np.eye(2)

        if FLAGS.prelearn_model_shape:
            self.stage = 2
        elif FLAGS.prelearn_model:
            self.stage = 1
        else:
            self.stage = 0

        # The format of labels will coordinates 0-1 correspond to position, 2 correspond to shape
        # 3 correspond color
        if self.stage == 0:
            # In the first stage have only generate cubes at each possible location
            # mask = (self.label[:, 0] < 0) & (self.label[:, 1] < 0)
            mask = (self.label[:, 2] == 0) & (self.label[:, 3] == 1)
            self.data = self.data[mask]
            self.label = self.label[mask]
        elif self.stage == 1:
            # In the second stage we learn to generate both cubes and colors and every possible location 
            # We restrict the mask to only to be cubes a particular quadrant
            # mask = (self.label[:, 2] == 0)
            mask = (self.label[:, 3] == 1)
            self.data = self.data[mask]
            self.label = self.label[mask]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        im = self.data[index] / 255.

        image_size = 64

        if FLAGS.datasource == 'default':
            im_corrupt = im + 0.3 * np.random.randn(image_size, image_size, 3)
        elif FLAGS.datasource == 'random':
            im_corrupt = 0.5 + 0.5 * np.random.randn(image_size, image_size, 3)

        if self.stage == 0:
            label_pos = self.label[index, :2]
            label = label_pos
        elif self.stage == 1:
            label_shape = self.shape_eye[int(self.label[index, 2])]
            label_pos = self.label[index, :2]

            label = (label_shape, label_shape, label_pos)
        elif self.stage == 2:
            label_color = self.color_eye[int(self.label[index, 3])]
            label_shape = self.shape_eye[int(self.label[index, 2])]
            label_pos = self.label[index, :2]

            label = (label_color, label_color, label_shape, label_pos)

        return im_corrupt, im, label


class CubesCrossProduct(Dataset):
    def __init__(self, ratio, cond_size=False, cond_pos=False, joint_baseline=False, inversion=False):
        dat = np.load("joint.npz")
        self.data = dat['ims']
        self.label = dat['labels']

        # Make sure different dual runs don't get different masks of objects
        np.random.seed(0)
        random.seed(0)

        sizes = np.unique(self.label[:, 2])
        x_vals = np.unique(self.label[:, 0])
        y_vals = np.unique(self.label[:, 1])
        total_list = list(product(x_vals, y_vals))
        # random.shuffle(total_list)

        if ratio != 1.0:
            stop_idx = int(ratio * len(total_list)) + 1
            select_idx = total_list[:stop_idx]
            reject_idx = total_list[stop_idx:]
            data_list = []
            label_list = []

            if not inversion:
                for tup in select_idx:
                    x, y = tup
                    mask = (self.label[:, 0] == x) & (self.label[:, 1] == y)

                    if not cond_pos:
                        data_list.append(self.data[mask])
                        label_list.append(self.label[mask])
                    else:
                        mask = (self.label[:, 0] == x) & (self.label[:, 1] == y) & (self.label[:, 2] ==  1.2)
                        data_list.append(self.data[mask])
                        label_list.append(self.label[mask])

            for tup in reject_idx:
                x, y = tup

                if not inversion:
                    mask = (self.label[:, 2] ==  1.2) & (self.label[:, 0] == x) & (self.label[:, 1] == y)
                else:
                    mask = (self.label[:, 2] !=  1.2) & (self.label[:, 0] == x) & (self.label[:, 1] == y)

                if not cond_size:
                    data_list.append(self.data[mask])
                    label_list.append(self.label[mask])

            self.data = np.concatenate(data_list, axis=0)
            self.label = np.concatenate(label_list, axis=0)


        self.cond_size = cond_size
        self.cond_pos = cond_pos
        self.joint_baseline = joint_baseline

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        im = self.data[index] / 255.

        image_size = 64

        if FLAGS.datasource == 'default':
            im_corrupt = im + 0.3 * np.random.randn(image_size, image_size, 3)
        elif FLAGS.datasource == 'random':
            im_corrupt = 0.5 + 0.5 * np.random.randn(image_size, image_size, 3)

        if self.cond_size:
            label = self.label[index, 2:3]
        elif self.cond_pos:
            label = self.label[index, :2]

        if self.joint_baseline:
            label = np.concatenate([self.label[index, 2:3], self.label[index, :2]])


        return im_corrupt, im, label


class CelebA(Dataset):

    def __init__(self):
        self.path = "/root/data/img_align_celeba"
        self.ims = os.listdir(self.path)
        self.ims = [osp.join(self.path, im) for im in self.ims]

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, index):
        label = 1

        if FLAGS.single:
            index = 0

        path = self.ims[index]
        im = imread(path)
        im = imresize(im, (32, 32))
        image_size = 32
        im = im / 255.

        if FLAGS.datasource == 'default':
            im_corrupt = im + 0.3 * np.random.randn(image_size, image_size, 3)
        elif FLAGS.datasource == 'random':
            im_corrupt = np.random.uniform(
                0, 1, size=(image_size, image_size, 3))

        return im_corrupt, im, label

class CelebA(Dataset):

    def __init__(self, cond_idx=1, filter_idx=0):
        self.path = "/datasets01_101/CelebA/072017/img_align_celeba"
        self.labels = pd.read_csv("list_attr_celeba.txt", sep="\s+", skiprows=1)
        self.cond_idx = cond_idx
        self.filter_idx = filter_idx

        if filter_idx != 0:
            mask = (self.labels.to_numpy()[:, self.cond_idx] == filter_idx)
            self.labels = self.labels[mask].reset_index()

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):


        if FLAGS.single:
            index = 0

        info = self.labels.iloc[index]
        if self.filter_idx != 0:
            fname = info['index']
        else:
            fname = info.name
        path = osp.join(self.path, fname)
        im = imread(path)
        im = imresize(im, (128, 128))
        image_size = 128
        im = im / 255.

        label = int(info.iloc[self.cond_idx])
        if label == -1:
            label = 0
        label = np.eye(2)[label]

        if FLAGS.datasource == 'default':
            im_corrupt = im + 0.3 * np.random.randn(image_size, image_size, 3)
        elif FLAGS.datasource == 'random':
            im_corrupt = np.random.uniform(
                0, 1, size=(image_size, image_size, 3))

        return im_corrupt, im, label
