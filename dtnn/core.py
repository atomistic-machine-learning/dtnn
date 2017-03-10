import os
import logging
import numpy as np

import tensorflow as tf
from dtnn.utils import shape


def batching(features, batch_size, num_batch_threads):
    in_names = list(features.keys())
    in_list = [features[name] for name in in_names]
    out_shapes = [shape(inpt) for inpt in in_list]

    features = tf.train.batch(
        in_list, batch_size, dynamic_pad=True,
        shapes=out_shapes, num_threads=num_batch_threads
    )
    features = dict(list(zip(in_names, features)))
    return features


class Calculator(object):
    pass


class Model(object):
    def __init__(self, model_dir, preprocessor_fcn=None, model_fcn=None,
                 **config):
        self.reuse = None
        self.preprocessor_fcn = preprocessor_fcn
        self.model_fcn = model_fcn

        self.model_dir = model_dir
        self.config = config
        self.config_path = os.path.join(self.model_dir, 'config.npz')

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if os.path.exists(self.config_path):
            logging.warning('Config file exists in model directory. ' +
                            'Config arguments will be overwritten!')
            self.from_config(self.config_path)
        else:
            self.to_config(self.config_path)

        with tf.variable_scope(None,
                               default_name=self.__class__.__name__) as scope:
            self.scope = scope
        self.saver = None

    def __getattr__(self, item):
        if item in list(self.config.keys()):
            return self.config[item]
        raise AttributeError

    def to_config(self, config_path):
        np.savez(config_path, **self.config)

    def from_config(self, config_path):
        cfg = np.load(config_path)
        for k, v in list(cfg.items()):
            if v.shape == ():
                v = v.item()
            self.config[k] = v

    def _preprocessor(self, features):
        if self.preprocessor_fcn is None:
            return features
        else:
            return self.preprocessor_fcn(features)

    def _model(self, features):
        if self.model_fcn is None:
            raise NotImplementedError
        else:
            return self.model_fcn(features)

    def init_model(self):
        pass

    def store(self, sess, iteration, name='best'):
        checkpoint_path = os.path.join(self.model_dir, name)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        if self.saver is None:
            raise ValueError('Saver is not initialized. ' +
                             'Build the model by calling `get_output`' +
                             'before storing it.')

        self.saver.save(sess, os.path.join(checkpoint_path, name), iteration)

    def restore(self, sess, name='best', iteration=None):
        checkpoint_path = os.path.join(self.model_dir, name)

        if not os.path.exists(checkpoint_path):
            return 0

        if self.saver is None:
            raise ValueError('Saver is not initialized. ' +
                             'Build the model by calling `get_output`' +
                             'before restoring it.')

        if iteration is None:
            chkpt = tf.train.latest_checkpoint(checkpoint_path)
        else:
            chkpt = os.path.join(checkpoint_path, name + '-' + str(iteration))
        logging.info('Restoring ' + chkpt)

        self.saver.restore(sess, chkpt)
        start_iter = int(chkpt.split('-')[-1])
        return start_iter

    def get_output(self, features, is_training, batch_size=None,
                   num_batch_threads=1):
        with tf.variable_scope(self.scope, reuse=self.reuse):
            with tf.variable_scope('preprocessing'):
                features = self._preprocessor(features)

            with tf.variable_scope('batching'):
                if batch_size is None:
                    features = {
                        k: tf.expand_dims(v, 0) for k, v in
                        list(features.items())
                        }
                else:
                    in_names = list(features.keys())
                    in_list = [features[name] for name in in_names]
                    out_shapes = [shape(inpt) for inpt in in_list]

                    features = tf.train.batch(
                        in_list, batch_size, dynamic_pad=True,
                        shapes=out_shapes, num_threads=num_batch_threads
                    )
                    features = dict(list(zip(in_names, features)))

            with tf.variable_scope('model'):
                self.init_model()
                features['is_training'] = is_training
                output = self._model(features)
                features.update(output)

        if self.saver is None:
            model_vars = [v for v in tf.global_variables()
                          if v.name.startswith(self.scope.name)]
            var_names = [v.name[len(self.scope.name):] for v in model_vars]
            vdict = dict(list(zip(var_names, model_vars)))
            self.saver = tf.train.Saver(vdict)
        self.reuse = True
        return features
