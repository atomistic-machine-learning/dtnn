
import os
import threading
from random import shuffle

import numpy as np
import tensorflow as tf
from ase.db import connect


def split_ase_db(asedb, dstdir, partitions, selection=None):
    partition_ids = list(partitions.keys())
    partitions = np.array(list(partitions.values()))
    if len(partitions[partitions < -1]) > 1:
        raise ValueError(
            'There must not be more than one partition of unknown size!')

    with connect(asedb) as con:
        ids = []
        for row in con.select(selection=selection):
            ids.append(row.id)

    ids = np.random.permutation(ids)
    n_rows = len(ids)

    r = (0. < partitions) * (partitions < 1.)
    partitions[r] *= n_rows
    partitions = partitions.astype(np.int)

    if np.any(partitions < 0):
        remaining = n_rows - np.sum(partitions[partitions > 0])
        partitions[partitions < 0] = remaining

    if len(partitions[partitions < 0]) > 1:
        raise ValueError(
            'Size of the partitions has to be <= the number of atom rows!')

    if not os.path.exists(dstdir):
        os.makedirs(dstdir)
    else:
        raise ValueError('Split destination directory already exists:',
                         dstdir)

    split_dict = {}
    with connect(asedb) as con:
        offset = 0
        if partition_ids is None:
            partition_ids = list(range(len(partitions)))
        for pid, p in zip(partition_ids, partitions):
            with connect(os.path.join(dstdir, pid + '.db')) as dstcon:
                print(offset, p)
                split_dict[pid] = ids[offset:offset + p]
                for i in ids[offset:offset + p]:
                    row = con.get(int(i))
                    if hasattr(row, 'data'):
                        data = row.data
                    else:
                        data = None
                    dstcon.write(row.toatoms(),
                                 key_value_pairs=row.key_value_pairs,
                                 data=data)
            offset += p
    np.savez(os.path.join(dstdir, 'split_ids.npz'), **split_dict)


data_feeds = []


def add_data_feed(data_feed):
    data_feeds.append(data_feed)


def start_data_feeds(sess, coord):
    for df in data_feeds:
        df.create_threads(sess, coord, True, True)


class LeapDataProvider(object):
    def __init__(self, batch_size=1):
        self._is_running = False
        self.batch_size = batch_size
        add_data_feed(self)

    @property
    def num_examples(self):
        raise NotImplementedError

    def get_features(self):
        raise NotImplementedError

    def get_property(self, pname):
        raise NotImplementedError

    def _run(self, sess, coord=None):
        raise NotImplementedError

    def create_threads(self, sess, coord=None, daemon=False, start=False):
        if self._is_running:
            return []

        thread = threading.Thread(target=self._run, args=(sess, coord))

        if daemon:
            thread.daemon = True
        if start:
            thread.start()

        self._is_running = True
        return [thread]


class ASEDataProvider(LeapDataProvider):
    def __init__(self, asedb, kvp={}, data={}, batch_size=1,
                 selection=None, shuffle=True, prefetch=False,
                 block_size=150000,
                 capacity=5000, num_epochs=np.Inf, floatX=np.float32):
        super(ASEDataProvider, self).__init__(batch_size)

        self.asedb = asedb
        self.prefetch = prefetch
        self.selection = selection
        self.block_size = block_size
        self.shuffle = shuffle
        self.kvp = kvp
        self.data = data
        self.floatX = floatX
        self.feat_names = ['numbers', 'positions', 'cell',
                           'pbc'] + list(kvp.keys()) + list(data.keys())
        self.shapes = [(None,), (None, 3), (3, 3),
                       (3,)] + list(kvp.values()) + list(data.values())

        self.epoch = 0
        self.num_epochs = num_epochs
        self.n_rows = 0

        # initialize queue
        with connect(self.asedb) as con:
            row = list(con.select(self.selection, limit=1))[0]

        feats = self.convert_atoms(row)
        dtypes = [np.array(feat).dtype for feat in feats]
        self.queue = tf.FIFOQueue(capacity, dtypes)

        self.placeholders = [
            tf.placeholder(dt, name=name)
            for dt, name in zip(dtypes, self.feat_names)
            ]
        self.enqueue_op = self.queue.enqueue(self.placeholders)
        self.dequeue_op = self.queue.dequeue()

        self.preprocs = []

    def convert_atoms(self, row):
        numbers = row.get('numbers')
        positions = row.get('positions').astype(self.floatX)
        pbc = row.get('pbc')
        cell = row.get('cell').astype(self.floatX)
        features = [numbers, positions, cell, pbc]

        for k in list(self.kvp.keys()):
            f = row[k]
            if np.isscalar(f):
                f = np.array([f])
            if f.dtype in [np.float16, np.float32, np.float64]:
                f = f.astype(self.floatX)
            features.append(f)
        for k in list(self.data.keys()):
            f = np.array(row.data[k])
            if np.isscalar(f):
                f = np.array([f])
            if f.dtype in [np.float16, np.float32, np.float64]:
                f = f.astype(self.floatX)
            features.append(f)
        return features

    def do_reload(self):
        with connect(self.asedb) as con:
            n_rows = con.count(self.selection)
        if self.n_rows != n_rows:
            self.n_rows = n_rows
            return True
        return False

    @property
    def num_examples(self):
        with connect(self.asedb) as con:
            n_rows = con.count(self.selection)
        return n_rows

    def iterate(self):
        # get data base size
        with connect(self.asedb) as con:
            n_rows = con.count(self.selection)
        if self.block_size is None:
            block_size = n_rows
        else:
            block_size = self.block_size
        n_blocks = int(np.ceil(n_rows / block_size))

        # shuffling
        if self.shuffle:
            permutation = np.random.permutation(n_blocks)
        else:
            permutation = range(n_blocks)

        # iterate over blocks
        for i in permutation:
            # load block
            with connect(self.asedb) as con:
                rows = list(
                    con.select(self.selection, limit=block_size,
                               offset=i * block_size)
                )

            # iterate over rows
            for row in rows:
                yield self.convert_atoms(row)
        self.epoch += 1

    def _run(self, sess, coord=None):
        while self.epoch < self.num_epochs:
            if self.prefetch:
                if self.do_reload():
                    data = []
                    with connect(self.asedb) as con:
                        for row in con.select(self.selection):
                            data.append(self.convert_atoms(row))
                if self.shuffle:
                    shuffle(data)
            else:
                data = self.iterate()

            for feats in data:
                fdict = dict(zip(self.placeholders, feats))
                sess.run(self.enqueue_op, feed_dict=fdict)

    def get_features(self):
        feat_dict = {}
        for name, feat, shape in zip(self.feat_names, self.dequeue_op,
                                     self.shapes):
            feat.set_shape(shape)
            feat_dict[name] = feat

        for preproc in self.preprocs:
            preproc(feat_dict)
        return feat_dict

    def add_preprocessor(self, preproc):
        self.preprocs.append(preproc)

    def get_property(self, pname):
        props = []
        with connect(self.asedb) as con:
            for rows in con.select(self.selection):
                try:
                    p = rows[pname]
                except Exception:
                    p = rows.data[pname]
                props.append(p)
        return props
