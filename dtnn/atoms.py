import numpy as np
import tensorflow as tf


def interatomic_distances(positions, cell, pbc, cutoff):
    with tf.variable_scope('distance'):
        # calculate heights
        icell = tf.matrix_inverse(cell)
        height = 1. / tf.sqrt(tf.reduce_sum(tf.square(icell), 0))

        extent = tf.where(tf.cast(pbc, tf.bool),
                          tf.cast(tf.floor(cutoff / height), tf.int32),
                          tf.cast(tf.zeros_like(height), tf.int32))
        n_reps = tf.reduce_prod(2 * extent + 1)

        # replicate atoms
        r = tf.range(-extent[0], extent[0] + 1)
        v0 = tf.expand_dims(r, 1)
        v0 = tf.tile(v0,
                     tf.stack(((2 * extent[1] + 1) * (2 * extent[2] + 1), 1)))
        v0 = tf.reshape(v0, tf.stack((n_reps, 1)))

        r = tf.range(-extent[1], extent[1] + 1)
        v1 = tf.expand_dims(r, 1)
        v1 = tf.tile(v1, tf.stack((2 * extent[2] + 1, 2 * extent[0] + 1)))
        v1 = tf.reshape(v1, tf.stack((n_reps, 1)))

        v2 = tf.expand_dims(tf.range(-extent[2], extent[2] + 1), 1)
        v2 = tf.tile(v2,
                     tf.stack((1, (2 * extent[0] + 1) * (2 * extent[1] + 1))))
        v2 = tf.reshape(v2, tf.stack((n_reps, 1)))

        v = tf.cast(tf.concat((v0, v1, v2), axis=1), tf.float32)
        offset = tf.matmul(v, cell)
        offset = tf.expand_dims(offset, 0)

        # add axes
        positions = tf.expand_dims(positions, 1)
        rpos = positions + offset
        rpos = tf.expand_dims(rpos, 0)
        positions = tf.expand_dims(positions, 1)

        euclid_dist = tf.sqrt(
            tf.reduce_sum(tf.square(positions - rpos),
                          reduction_indices=3))
        return euclid_dist


def site_rdf(distances, cutoff, step, width, eps=1e-5,
             use_mean=False, lower_cutoff=None):
    with tf.variable_scope('srdf'):
        if lower_cutoff is None:
            vrange = cutoff
        else:
            vrange = cutoff - lower_cutoff
        distances = tf.expand_dims(distances, -1)
        n_centers = np.ceil(vrange / step)
        gap = vrange - n_centers * step
        n_centers = int(n_centers)

        if lower_cutoff is None:
            centers = tf.linspace(0., cutoff - gap, n_centers)
        else:
            centers = tf.linspace(lower_cutoff + 0.5 * gap, cutoff - 0.5 * gap,
                                  n_centers)
        centers = tf.reshape(centers, (1, 1, 1, -1))

        gamma = -0.5 / width / step ** 2

        rdf = tf.exp(gamma * (distances - centers) ** 2)

        mask = tf.cast(distances >= eps, tf.float32)
        rdf *= mask
        rdf = tf.reduce_sum(rdf, 2)
        if use_mean:
            N = tf.reduce_sum(mask, 2)
            N = tf.maximum(N, 1)
            rdf /= N

        new_shape = [None, None, n_centers]
        rdf.set_shape(new_shape)

    return rdf
