import tensorflow as tf


class CostFunction(object):
    def __init__(self, name):
        self.name = name

    def __call__(self, output):
        errors = self.calc_errors(output)
        return self.aggregate(errors)

    def calc_errors(self, output):
        raise NotImplementedError

    def aggregate(self, errors):
        return tf.reduce_mean(errors)


class MeanAbsoluteError(CostFunction):
    def __init__(self, prediction, target, idx=None, name='MAE'):
        super(MeanAbsoluteError, self).__init__(name)
        self.prediction = prediction
        self.target = target
        self.idx = idx

    def calc_errors(self, output):
        tgt = output[self.target]
        pred = output[self.prediction]
        if self.idx is not None:
            tgt = tgt[:, self.idx]
            pred = pred[:, self.idx]
        return tf.abs(tgt - pred)


class L2Loss(CostFunction):
    def __init__(self, prediction, target, idx=None, name='MSE'):
        super(L2Loss, self).__init__(name)
        self.prediction = prediction
        self.target = target
        self.idx = idx

    def calc_errors(self, output):
        tgt = output[self.target]
        pred = output[self.prediction]
        if self.idx is not None:
            tgt = tgt[:, self.idx]
            pred = pred[:, self.idx]
        return (tgt - pred) ** 2

    def aggregate(self, errors):
        return tf.reduce_sum(errors)


class MeanSquaredError(CostFunction):
    def __init__(self, prediction, target, idx=None, name='MSE'):
        super(MeanSquaredError, self).__init__(name)
        self.prediction = prediction
        self.target = target
        self.idx = idx

    def calc_errors(self, output):
        tgt = output[self.target]
        pred = output[self.prediction]
        if self.idx is not None:
            tgt = tgt[:, self.idx]
            pred = pred[:, self.idx]
        return (tgt - pred) ** 2


class RootMeanSquaredError(MeanSquaredError):
    def __init__(self, prediction, target, idx=None, name='RMSE'):
        super(RootMeanSquaredError, self).__init__(prediction, target, idx,
                                                   name)

    def aggregate(self, errors):
        return tf.sqrt(tf.reduce_mean(errors))


class PAMeanAbsoluteError(CostFunction):
    def __init__(self, prediction, target, idx=None, name='MAE'):
        super(PAMeanAbsoluteError, self).__init__(name)
        self.prediction = prediction
        self.target = target
        self.idx = idx

    def calc_errors(self, output):
        Z = output['numbers']
        N = tf.reduce_sum(tf.cast(tf.greater(Z, 0), tf.float32), 1)

        tgt = output[self.target]
        pred = output[self.prediction]
        if self.idx is not None:
            tgt = tgt[:, self.idx]
            pred = pred[:, self.idx]
        return tf.abs(tgt - pred) / N


class PARmse(CostFunction):
    def __init__(self, prediction, target, idx=None, name='MSE'):
        super(PARmse, self).__init__(name)
        self.prediction = prediction
        self.target = target
        self.idx = idx

    def calc_errors(self, output):
        Z = output['numbers']
        N = tf.reduce_sum(tf.cast(tf.greater(Z, 0), tf.float32), 1)

        tgt = output[self.target]
        pred = output[self.prediction]
        if self.idx is not None:
            tgt = tgt[:, self.idx]
            pred = pred[:, self.idx]
        return ((tgt - pred) / N) ** 2

    def aggregate(self, errors):
        return tf.sqrt(tf.reduce_mean(errors))
