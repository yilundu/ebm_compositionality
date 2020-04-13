import collections
import numpy as np
from multiprocessing import Pool
import multiprocessing
import tensorflow as tf
import random
import pickle, sys, time
import matplotlib.pyplot as plt
import argparse
import time
from scipy.misc import imsave
from pygame.color import THECOLORS

chunksize = 1

def get_sim(shape_infos, camera_info):
    from mujoco_py import load_model_from_xml, MjSim
    xml = """
        <mujoco>
        <!--
        <default>
            <geom material="DEFAULT_GEOM_MAT"/>
        </default>
        -->

        <asset>
            <texture name="SKYBOX" type="skybox" builtin="gradient" width="128" height="128" rgb1="0.4 0.6 0.8" rgb2="0 0 0"/>
            <texture name="DEFAULT_GEOM_TEX" type="cube" builtin="flat" mark="cross" width="64" height="64" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" markrgb="1 1 1" random="0.01"/>
            <material name="DEFAULT_GEOM_MAT" texture="DEFAULT_GEOM_TEX" texuniform="false"/>
        </asset>

        <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <camera name="main" pos="{0} {1} {2}" fovy="45" mode="targetbody" target="plane"/>

        <body name="plane">
            <geom type="box" size="40 40 0.5" rgba=".9 0 0 1" pos="0 0 -0.5"/>
        </body>
        """.format(camera_info[0], camera_info[1], camera_info[2])

    for i, shape_info in enumerate(shape_infos):

        tx = shape_info['tx']
        ty = shape_info['ty']
        tz = shape_info['tz']

        sx = shape_info['sx']
        sy = shape_info['sy']
        sz = shape_info["sz"]
        r, g, b, a = shape_info['rgba']
        type_cube = shape_info['type']

        xml += """
            <body name="cube_{6}">
                <joint axis="1 0 0" type="hinge"/>
                <joint axis="0 1 0" type="hinge"/>
                <joint axis="0 0 1" type="hinge"/>
                <geom pos="{0} {1} {2}" size="{3} {4} {5}" type="{7}" material="DEFAULT_GEOM_MAT" rgba="{8} {9} {10} {11}"/>
            </body>
            """.format(tx, ty, tz, sx, sy, sz, i, type_cube, r, g, b, a)


    xml += """
           </worldbody>
           </mujoco>
           """

    model = load_model_from_xml(xml)
    sim = MjSim(model)
    return sim


def generate_random_pose():
    z = np.random.uniform(0.6, 1.7)
    y = np.random.uniform(0, 1)

    if y > 0.5:
        y = (y - 0.5) * 2 * 1 + 1.5
    else:
        y = (y - 0.5) * 2 * 1 - 1.5


    x = np.random.uniform(-0.4, 0.4)

    return [x, y, z]


def generate_triple_scene():
    # Generate red, blue and green squares of slightly varying sizes at different 
    # positions so that they don't overlap

    # Items will be generate in a [-1, 1] region:

    def is_valid(cond_pos, size, infos):
        for info in infos:
            old_pos, old_size = info

            if (np.abs(cond_pos - old_pos) < (size + old_size)).any():
                return False

        return True

    sizes = np.random.uniform(0.1, 0.2, (3))
    colors = [(0.97, 0.97, 0.164, 0.7), (0, 1, 0, 0.7), (0, 0, 1, 0.7)]

    infos = []
    shape_infos = []
    encode_dat = []
    for i in range(3):
        cand_pos = np.random.uniform(-1, 1, 2)
        size = sizes[i]

        for j in range(1000):
            if not is_valid(cand_pos, size, infos):
                cand_pos = np.random.uniform(-1, 1, 2)
                size = sizes[i]
            else:
                break

            if j == 999:
                print('ERROR!!!')

        infos.append((cand_pos, size))

        shape_info = {"tx": cand_pos[0], "ty": cand_pos[1], "tz": size, "sx": size, "sy": size, "sz": size, "rgba": colors[i], "type": "box"}
        shape_infos.append(shape_info)

        encode_dat.extend(list(cand_pos) + [size])


    ims = []
    for i in range(3):
        camera_info = generate_random_pose()
        sim = get_sim(shape_infos, camera_info)
        sim.forward()
        im = sim.render(64, 64, camera_name="main")

        im = im[::-1, :, :].copy()
        encode_dat.extend(camera_info)
        ims.append(im)

    ims = np.array(ims)
    # print(ims.max(), ims.min(), ims.std())

    encode_dat = np.array(encode_dat)

    return ims, encode_dat


def generate_whole_sim(rank):
    print("Starting rank {}".format(rank))
    random.seed(rank)
    np.random.seed(rank)

    n = random.randint(1, 4)
    idxs = random.sample(list(range(4)), n)

    labels_total, ims = [], []

    for i in range(chunksize):
        im, label_numpy = generate_triple_scene()

        labels_total.append(label_numpy)
        ims.append(im)

    im = np.stack(ims, axis=0)
    label = np.stack(labels_total, axis=0)
    print("Finished Processing ", im.shape, label.shape)

    return im.astype(np.uint8), label


def generate_whole_sim_timeout(rank):
    return generate_whole_sim(rank)

# pool = multiprocessing.pool.ThreadPool(1)
# return pool.apply_async(generate_whole_sim, [rank]).get(timeout=20)


def generate_sims(traj_num=100000):

    chunk = 1000

    n = traj_num // chunk
    total_ims = []
    total_labels = []

    for i in range(n):
        args = list(range(i*chunk, (i+1)*chunk))
        pool = Pool()
        results = pool.map(generate_whole_sim_timeout, args)
        ims, labels = zip(*results)
        ims = np.concatenate(ims, axis=0).astype(np.uint8)
        labels = np.concatenate(labels, axis=0)

        total_ims.append(ims)
        total_labels.append(labels)

    ims = np.concatenate(total_ims, axis=0).astype(np.uint8)
    labels = np.concatenate(total_labels, axis=0)

    np.savez("/root/cubes_triple_view.npz", ims=ims, labels=labels)


if __name__ == "__main__":
    # How is a neural network is able to match stuff

    # TODO decribe how the fuck camera pose varies
    # Given values x, y, z
    # It appears z is the height so probably around (3-7)
    # It apepars y represents the forward and back distance cubes so should be outside ty
    # Somewhat unintuitively -y corresponds to close to the camera while y corresponds to away from camera

    # camera_info = [3, -3, 7.5]

    # shape_infos = [{"tx": 0.1, "ty": -0.1, "tz": 0.3, "sx": 0.3, "sy": 0.3, "sz": 0.3, "rgba": (0, 1, 0, 1), "type": "box"}]

    # ims, encode_dat = generate_triple_scene()

    # for i in range(3):
    #     imsave("debug{}.png".format(i), ims[i])

    generate_sims(100000)
    # generate_whole_sim(chunksize)
