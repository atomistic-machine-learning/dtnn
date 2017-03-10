#!/usr/bin/env python
"""
    Example script for evaluation a DTNN to predict
    the total energy at 0K (U0) for the GDB-9 data.
"""
import argparse
import os

import numpy as np
import tensorflow as tf
from ase.db import connect

from dtnn.models import DTNN


def evaluate(args):
    # define model inputs
    features = {
        'numbers': tf.placeholder(tf.int64, shape=(None,)),
        'positions': tf.placeholder(tf.float32, shape=(None, 3)),
        'cell': np.eye(3).astype(np.float32),
        'pbc': np.zeros((3,)).astype(np.int64)
    }

    # load model
    model = DTNN(args.model_dir)
    model_output = model.get_output(features, is_training=False)
    y = model_output['y']

    with tf.Session() as sess:
        model.restore(sess)
        
        print('test_live.db')
        U0_live, U0_pred_live = predict(
            os.path.join(args.split_dir, 'test_live.db'), features, sess, y
        )
        print('test.db')
        U0, U0_pred = predict(
            os.path.join(args.split_dir, 'test.db'), features, sess, y
        )
        U0 += U0_live
        U0_pred += U0_pred_live
        U0 = np.vstack(U0)
        U0_pred = np.vstack(U0_pred)

        diff = U0 - U0_pred
        mae = np.mean(np.abs(diff))
        rmse = np.sqrt(np.mean(diff ** 2))
        print('MAE: %.3f eV, RMSE: %.3f eV' % (mae, rmse))


def predict(dbpath, features, sess, y):
    U0 = []
    U0_pred = []
    count = 0
    with connect(dbpath) as conn:
        n_structures = conn.count()
        for row in conn.select():
            U0.append(row['U0'])

            at = row.toatoms()
            feed_dict = {
                features['numbers']:
                    np.array(at.numbers).astype(np.int64),
                features['positions']:
                    np.array(at.positions).astype(np.float32)
            }
            U0_p = sess.run(y, feed_dict=feed_dict)
            U0_pred.append(U0_p)
            if count % 1000 == 0:
                print(str(count) + ' / ' + str(n_structures))
            count += 1
    return U0, U0_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('split_dir',
                        help='Path to directory with data splits' +
                             ' ("test.db", "test_live.db")')
    parser.add_argument('model_dir', help='Path to model directory.')
    args = parser.parse_args()

    evaluate(args)
