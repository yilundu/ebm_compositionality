import collections
import numpy as np
from multiprocessing import Pool
import tensorflow as tf
import random
import pickle, sys, time
import matplotlib.pyplot as plt
import argparse
import time
from scipy.misc import imsave
from pygame.color import THECOLORS

chosen_colors = list(THECOLORS.keys())[:10]
chunksize = 20

def get_sim(cube_infos):
    from mujoco_py import load_model_from_xml, MjSim
    # generate random lighting
    light = np.random.uniform(0, 1, (3,))

    random_perturb = np.random.uniform(-0.1, 0.1, (4,))
    plane_color = np.clip(np.array([.9, 0, 0, 1.0]) + random_perturb, 0, 1)
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
        <light diffuse="{0} {1} {2}" pos="0 0 3" dir="0 0 -1"/>
        <camera name="main" pos="0 3.5 {7}" fovy="45" mode="targetbody" target="plane"/>

        <body name="plane">
            <geom type="box" size="5 5 0.05" rgba="{3} {4} {5} {6}" pos="0 0 -0.05"/>
        </body>
        """.format(*light, *plane_color, np.random.uniform(2.5, 4.0))

    for i, cube_info in enumerate(cube_infos):

        tx = cube_info['tx']
        ty = cube_info['ty']
        tz = cube_info['tz']

        sx = cube_info['sx']
        sy = cube_info['sy']
        sz = cube_info["sz"]
        rgba = cube_info['rgba']
        type_cube = cube_info['type']
        color = np.array((0, 1, 0, 0.7))
        random_perturb = np.random.uniform(-0.2, 0.2, (4,))
        plane_color = np.clip(color + random_perturb, 0, 1)

        if type_cube == "box":
            xml += """
                <body name="cube_{6}">
                    <geom pos="{0} {1} {2}" size="{3} {4} {5}" type="{7}" material="DEFAULT_GEOM_MAT"rgba="{8} {9} {10} {11}"/>
                </body>
                """.format(tx, ty, tz, sx, sy, sz+0.01, i, type_cube, *rgba)
        elif type_cube == "sphere":
            xml += """
                <body name="cube_{6}">
                    <geom pos="{0} {1} {2}" size="{5}" type="{7}" material="DEFAULT_GEOM_MAT"rgba="{8} {9} {10} {11}"/>
                </body>
                """.format(tx, ty, tz, sx, sy, sz+0.01, i, type_cube, *rgba)
        elif type_cube == "cylinder":
            xml += """
                <body name="cube_{6}">
                    <geom pos="{0} {1} {2}" size="{4} {5}" type="{7}" material="DEFAULT_GEOM_MAT"rgba="{8} {9} {10} {11}"/>
                </body>
                """.format(tx, ty, tz, sx, sy, sz+0.01, i, type_cube, *rgba)


    xml += """
           </worldbody>
           </mujoco>
           """

    model = load_model_from_xml(xml)
    sim = MjSim(model)
    return sim


def generate_quadrant(idx):
    # Range between 0 to 1.5
    quads = [(-0.75, -0.75), (-0.75, 0.75), (0.75, 0.75), (0.75, -0.75)]
    dx, dy = np.random.uniform(-0.2, 0.2, (2,))
    size = np.random.uniform(0.4, 0.5)
    color = random.choice(chosen_colors)
    rgba = [c / 255. for c in list(THECOLORS[color])]

    s = quads[idx]
    types = ["box", "sphere", "cylinder"]

    cubes = {'tx': dx, 'ty': dy, 'tz': size, 'sx': size, 'sy': size, 'sz': size, 'rgba': rgba, 'type': random.choice(types)}

    return cubes


def generate_single():
    # Range between 0 to 1.5
    dx, dy = np.random.uniform(-1.0, 1.0, (2,))
    size = np.random.uniform(0.3, 0.5)

    color = np.array((0, 1, 0, 0.7))
    random_perturb = np.random.uniform(-0.2, 0.2, (4,))
    rgba = np.clip(color + random_perturb, 0, 1)

    s = (0.0, 0.0)
    types = ["box", "sphere", "cylinder"]

    cubes = {'tx': dx, 'ty': dy, 'tz': size, 'sx': size, 'sy': size, 'sz': size, 'rgba': rgba, 'type': random.choice(types)}

    return cubes

def generate_junk(labels):
    # Generate a bunch of trash to confuse model
    label = labels[0]
    s, x, y = label['sz'], label['tx'], label['ty']

    valid_list = [(x - s, x + s, y - s, y + s)]
    types = ["box", "sphere", "cylinder"]



    for i in range(5):
        # Try to generate 5 pieces of chunch
        colors = [np.array((0, 1, 0, 0.7)), np.array((0, 0, 1, 0.7)), np.array((153/255, 51/255, 1, 0.7))]
        color = random.choice(colors)
        random_perturb = np.random.uniform(-0.2, 0.2, (4,))
        rgba = np.clip(color + random_perturb, 0, 1)

        for t in range(10):
            dx, dy = np.random.uniform(-1.0, 1.0, (2,))
            size = np.random.uniform(0.3, 0.5)

            status = True
            for v in valid_list:
                if dx - size > v[1] or size + dx < v[0] or dy - size > v[3] or dy + size < v[2]:
                    pass
                else:
                    status = False
                    break

            if status is True:
                cubes = {'tx': dx, 'ty': dy, 'tz': size, 'sx': size, 'sy': size, 'sz': size, 'rgba': rgba, 'type': random.choice(types)}
                labels.append(cubes)
                valid_list.append((dx - size, dx + size, dy - size, dy + size))
                break

    return labels


def generate_default_sim():
    # labels = [generate_quadrant(i) for i in range(4)]
    # print(len(labels))
    labels = [generate_single()]
    label_numpy = np.zeros(24)

    for idx, label in enumerate(labels):
        label_numpy[idx*6:idx*6+6] = [label['tx'], label['ty'], label['tz'], label['sx'], label['sy'], label['sz']]

    labels_new = generate_junk(labels)

    sim = get_sim(labels_new)
    sim.forward()
    im = sim.render(64, 64, camera_name="main", depth=False)

    im_channel = im[:, :, 1]
    if (im_channel > 60).any():
        return im, label_numpy
    else:
        print("Error no image:  ")
        return generate_default_sim()


def generate_pair_sim():
    # Generate cubes dataset where there are pairs of items

    # Labels
    # Format:
    #       First entry denotes near(1) or far, followed by tx, ty, tz, sx, sy, sz
    #       and shape (0, 1, 2) for cube, sphere, cylinder respectively

    n_obj = 3
    near = random.choice([0, 1])
    types = ["box", "sphere", "cylinder"]

    if near:
        ts = [(-0.4, 0.4), (0.4, 0.4), (0.0 , -0.2)]
    else:
        ts = [(-0.9, 0.2), (0.9, 0.2), (0.0, -0.2)]

    labels = []
    label_numpy = np.zeros(15)
    label_numpy[0] = near

    for i, t in enumerate(ts):

        if np.random.uniform() > 0.5:
            info = {}

            size = random.choice([0.2, 0.3])
            dx, dy = np.random.uniform(-0.1, 0.1, (2,))
            shape_idx = random.choice(list(range(3)))
            shape_type = types[shape_idx]

            if shape_type == "sphere":
                size = size / 2.

            label_numpy[i*7+1:i*7+2] = shape_idx
            label_numpy[i*7+2:i*7+5] = [t[0] - dx, t[1] - dy, size]
            label_numpy[i*7+5:i*7+8] = size


            color = random.choice(chosen_colors)
            rgba = [c / 255. for c in list(THECOLORS[color])]
            cubes = {'tx': t[0] - dx, 'ty': t[1] - dy, 'tz': size, 'sx': size, 'sy': size, 'sz': size, 'rgba': rgba, 'type': shape_type}
            labels.append(cubes)

    sim = get_sim(labels)

    im = sim.render(64, 64, camera_name="main")

    return im, label_numpy


def generate_whole_sim(rank, near=False):
    print("Processing rank {}".format(rank))
    random.seed(rank)

    n = random.randint(1, 4)
    idxs = random.sample(list(range(4)), n)

    labels_total, ims = [], []

    for i in range(chunksize):
        if near:
            im, label_numpy = generate_pair_sim()
        else:
            im, label_numpy = generate_default_sim()

        labels_total.append(label_numpy)
        ims.append(im)

    im = np.stack(ims, axis=0)
    label = np.stack(labels_total, axis=0)

    return im, label


def near_fn(inp):
    return generate_whole_sim(inp, near=True)

def generate_sims(traj_num=100000):
    args = list(range(max(traj_num // chunksize, 1)))
    pool = Pool()

    results = pool.map(generate_whole_sim, args)
    ims, labels = zip(*results)
    ims = np.concatenate(ims, axis=0)
    labels = np.concatenate(labels, axis=0)

    print("Saving...")
    np.savez("cubes_varied_junk_801.npz", ims=ims, labels=labels)


if __name__ == "__main__":

    # cube_1 = {'tx': -1.5, 'ty': 0.0, 'tz': 0.25, 'sx': 0.25, 'sy': 0.25, 'sz': 0.25}
    # cube_2 = {'tx': 0.3, 'ty': -1.5, 'tz': 0.5, 'sx': 0.25, 'sy': 0.25, 'sz': 0.5}
    # sim = get_sim([cube_1, cube_2])

    # seed = 0
    # random.seed(seed)
    # np.random.seed(seed)
    # im, _ = generate_default_sim()

    # im, _ = sim.render(64, 64, camera_name="main")
    # imsave("image.png", im[::-1, :])

    generate_sims(200000)

