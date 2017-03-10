""" Training procedures for machine learning models """

import os
import logging

import numpy as np
import tensorflow as tf

import dtnn


def early_stopping(model, cost_fcn, optimizer,
                   train_data, val_data, test_data=None,
                   additional_cost_fcns=[], global_step=None,
                   n_iterations=1000000, patience=float('inf'),
                   checkpoint_interval=100000, summary_interval=1000,
                   validation_interval=1000, coord=None,
                   num_val_batches=1, num_test_batches=1,
                   profile=False, session_config=None):
    """
    Train model using early stopping with validation and test set.

    :param LeapModel model: The model to be trained.
    :param CostFunction cost_fcn: Cost function to be optimized.
    :param LeapDataProvider train_data: Training data provider.
    :param LeapDataProvider val_data: Validation data provider.
    :param LeapDataProvider test_data:
        Test data provider for estimating the error during training. (optional)
    :param tf.train.Optimizer optimizer:
        Tensorflow optimizer (e.g. SGD, Adam, ...)
    :param list(CostFunction) additional_cost_fcns:
        List of additional cost functions for monitoring
    :param tf.Variable global_step:
        Variable containing the global step.
        Pass if using learning rate decay (optional)
    :param int n_iterations: Number of optimizer steps.
    :param int patience: Stop after `patience` steps without improved
                         validation cost. (optional)
    :param int checkpoint_interval: Save model with given frequency.
    :param int summary_interval: Store training summary with given frequency.
    :param int validation_interval: Validate model with given frequency.
    :param tf.train.Coordinator coord: Coordinator for threads.
    :param int num_val_batches: Iterate over `num_val_batches`
                                number of batches for validation.
    :param int num_test_batches: Iterate over `num_val_batches`
                                 number of batches for testing.
    :param bool profile: If `true`, enable collection of runtime data in
        TensorFlow
    """

    train_features = train_data.get_features()
    val_features = val_data.get_features()
    test_features = test_data.get_features()

    # retrieve model outputs
    train_output = model.get_output(
        train_features,
        is_training=True,
        batch_size=train_data.batch_size,
        num_batch_threads=4
    )
    val_output = model.get_output(
        val_features,
        is_training=False,
        batch_size=val_data.batch_size,
        num_batch_threads=1
    )
    test_output = model.get_output(
        test_features,
        is_training=False,
        batch_size=test_data.batch_size,
        num_batch_threads=1
    )

    # assemble costs & summaries
    cost = cost_fcn(train_output)
    train_sums = [
        tf.summary.scalar(add_cost_fcn.name,
                          add_cost_fcn(train_output))
        for add_cost_fcn in additional_cost_fcns
        ]
    train_sums.append(tf.summary.scalar('cost', cost))
    train_summaries = tf.summary.merge_all()

    # validation
    val_errors = [cost_fcn(val_output)]
    val_errors += [add_cost_fcn.calc_errors(val_output)
                   for add_cost_fcn in additional_cost_fcns]

    # test
    if test_output is not None:
        test_errors = [cost_fcn(test_output)]
        test_errors += [add_cost_fcn.calc_errors(test_output)
                        for add_cost_fcn in additional_cost_fcns]
    else:
        test_errors = None

    # collect test & validation summaries
    errors = [tf.placeholder(tf.float32) for _ in val_errors]
    summaries = [
        tf.summary.scalar(add_cost_fcn.name,
                          add_cost_fcn.aggregate(
                              err))
        for add_cost_fcn, err in
        zip(additional_cost_fcns, errors)
        ]
    summaries = tf.summary.merge(summaries)

    # training ops
    if global_step is None:
        global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(cost, global_step=global_step)
    init_op = tf.global_variables_initializer()

    # training loop
    best_error = np.Inf
    coord = tf.train.Coordinator() if coord is None else coord
    try:
        with tf.Session(config=session_config) as sess:
            # set up summary writers
            train_writer = tf.summary.FileWriter(
                os.path.join(model.model_dir, 'train'), sess.graph)
            val_writer = tf.summary.FileWriter(
                os.path.join(model.model_dir, 'validation'))
            test_writer = tf.summary.FileWriter(
                os.path.join(model.model_dir, 'test'))

            # initialize all variables
            sess.run(init_op)

            # setup Saver & restore if previous checkpoints are available
            chkpt_saver = tf.train.Saver()
            checkpoint_path = os.path.join(model.model_dir, 'chkpoints')
            if not os.path.exists(checkpoint_path):
                start_iter = 0
                os.makedirs(checkpoint_path)
            else:
                chkpt = tf.train.latest_checkpoint(checkpoint_path)
                chkpt_saver.restore(sess, chkpt)
                start_iter = int(chkpt.split('-')[-1])
            chkpt = os.path.join(checkpoint_path, 'checkpoint')

            global_step.assign(start_iter).eval()

            dtnn.data.start_data_feeds(sess, coord)
            tf.train.start_queue_runners(sess=sess, coord=coord)
            logging.info('Starting at iteration ' +
                         str(start_iter) + ' / ' + str(n_iterations))
            last_best = start_iter
            for i in range(start_iter, n_iterations):
                if i % checkpoint_interval == 0:
                    chkpt_saver.save(sess, chkpt, i)
                    logging.info('Saved checkpoint at iteration %d / %d',
                                 i, n_iterations)

                if (i - last_best) > patience:
                    logging.info('Out of patience.')
                    break

                if i % summary_interval == 0:
                    logging.debug('Store Summary.')
                    if profile:
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        _, train_sums = sess.run(
                            [train_op, train_summaries], options=run_options, run_metadata=run_metadata)
                        train_writer.add_run_metadata(run_metadata, 'step%d' % i)
                    else:
                        _, train_sums = sess.run([train_op, train_summaries])
                    train_writer.add_summary(train_sums, global_step=i)
                else:
                    sess.run(train_op)

                if i % validation_interval == 0:
                    val_costs = []
                    sums = [[] for _ in range(len(additional_cost_fcns))]
                    for k in range(num_val_batches):
                        results = sess.run(val_errors)
                        val_costs.append(results[0])
                        for s, r in enumerate(results[1:]):
                            sums[s].append(r)
                    for s in range(len(sums)):
                        sums[s] = np.vstack(sums[s])
                    val_cost = np.mean(val_costs)

                    feed_dict = {
                        err: vsum
                        for err, vsum in zip(errors, sums)
                    }
                    val_sums = sess.run(summaries, feed_dict=feed_dict)
                    val_writer.add_summary(val_sums,
                                           global_step=i)

                    if val_cost < best_error:
                        last_best = i
                        best_error = val_cost

                        test_costs = []
                        sums = [[] for _ in range(len(additional_cost_fcns))]
                        for k in range(num_test_batches):
                            results = sess.run(test_errors)
                            test_costs.append(results[0])
                            for s, r in enumerate(results[1:]):
                                sums[s].append(r)
                        test_cost = np.mean(test_costs)
                        for s in range(len(sums)):
                            sums[s] = np.vstack(sums[s])

                        feed_dict = {
                            err: vsum
                            for err, vsum in zip(errors, sums)
                            }
                        test_sums = sess.run(summaries, feed_dict=feed_dict)

                        test_writer.add_summary(test_sums,
                                                global_step=i)
                        model.store(sess, i, 'best')
                        logging.info(
                            'New best model at iteration %d /' +
                            ' %d with loss %.2f',
                            i, n_iterations, test_cost
                        )
            logging.info('Done')
    finally:
        logging.info('Saving chkpoint...')
        chkpt_saver.save(sess, chkpt, i)
        logging.info('Done.')

        logging.info('Stopping threads...')
        if coord is not None:
            coord.request_stop()
        logging.info('Done.')


