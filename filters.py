import tensorflow as tf
import numpy as np

stride_3 = np.array([1, 2, 1])
stride_5 = np.array([1, 4, 6, 4, 1])

stride_3 = stride_3[:, None] * stride_3[None, :]
stride_5 = stride_5[:, None] * stride_5[None, :]

stride_3 = stride_3 / stride_3.sum()
stride_5 = stride_5 / stride_5.sum()

stride_3 = stride_3[:, :, None, None]
stride_5 = stride_5[:, :, None, None]

stride_3 = tf.constant(stride_3, dtype=tf.float32)
stride_5 = tf.constant(stride_5, dtype=tf.float32)
