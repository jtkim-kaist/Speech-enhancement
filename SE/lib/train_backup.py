import tensorflow as tf
import trnmodel
import datareader2 as dr
import config
import os
import numpy as np
import time

from timeit import default_timer
from contextlib import contextmanager


@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start


def do_validation(m_valid, sess, valid_path):

    # dataset reader setting #

    valid_dr = dr.DataReader(valid_path["valid_input_path"], valid_path["valid_output_path"],
                             valid_path["norm_path"], dist_num=config.dist_num, is_shuffle=False, is_val=True)

    avg_valid_cost = 0.
    itr_sum = 0.

    cost_list = [0 for i in range(valid_dr._file_len)]
    itr_file = 0

    while True:

        # valid_inputs, valid_labels = valid_dr.next_batch(config.batch_size)
        valid_inputs, valid_labels = valid_dr.next_batch(config.batch_size)

        feed_dict = {m_valid.inputs: valid_inputs, m_valid.labels: valid_labels, m_valid.keep_prob: 1.0}

        # valid_cost, valid_softpred, valid_raw_labels\
        #     = sess.run([m_valid.cost, m_valid.softpred, m_valid.raw_labels], feed_dict=feed_dict)
        #
        # fpr, tpr, thresholds = metrics.roc_curve(valid_raw_labels, valid_softpred, pos_label=1)
        # valid_auc = metrics.auc(fpr, tpr)

        valid_cost = sess.run(m_valid.raw_cost, feed_dict=feed_dict)
        print(valid_cost)
        avg_valid_cost += valid_cost
        itr_sum += 1

        if valid_dr.file_change_checker():
            # print(itr_file)
            cost_list[itr_file] = avg_valid_cost / itr_sum
            avg_valid_cost = 0.
            itr_sum = 0
            itr_file += 1
            valid_dr.file_change_initialize()
            if valid_dr.eof_checker():
                valid_dr.reader_initialize()
                print('Valid data reader was initialized!')  # initialize eof flag & num_file & start index
                break

    total_avg_valid_cost = np.asscalar(np.mean(np.asarray(cost_list)))

    return total_avg_valid_cost


def main(argv=None):

    assert not (argv is None), "The project path must be provided."

    # set train path

    train_input_path = argv + '/data/train/noisy'
    train_output_path = argv + '/data/train/clean'
    norm_path = argv + '/data/train/norm'

    # set valid path
    valid_input_path = argv + '/data/valid/noisy'
    valid_output_path = argv + '/data/valid/clean'
    logs_dir = argv + '/logs'
    #                               Graph Part                               #

    print("Graph initialization...")

    global_step = tf.Variable(0, trainable=False)

    with tf.device(config.device):
        with tf.variable_scope("model", reuse=None):
            m_train = trnmodel.Model(is_training=True, global_step=global_step)
        with tf.variable_scope("model", reuse=True):
            m_valid = trnmodel.Model(is_training=False, global_step=global_step)

    print("Done")

    #                               Summary Part                             #

    tensor_names = [n.name for n in tf.get_default_graph().as_graph_def().node]

    with open(logs_dir + "/tensor_names.txt", 'w') as f:
        for t_name in tensor_names:
            f.write("%s\n" % str(t_name))

    print("Setting up summary op...")

    summary_ph = tf.placeholder(dtype=tf.float32)
    with tf.variable_scope("Training_procedure"):

        cost_summary_op = tf.summary.scalar("cost", summary_ph)

    print("Done")

    #                               Model Save Part                           #

    print("Setting up Saver...")
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    print("Done")

    #                               Session Part                              #

    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    train_summary_writer = tf.summary.FileWriter(logs_dir + '/train/', sess.graph, max_queue=2)
    valid_summary_writer = tf.summary.FileWriter(logs_dir + '/valid/', max_queue=2)

    if ckpt and ckpt.model_checkpoint_path:  # model restore

        print("Model restored...")

        saver.restore(sess, ckpt.model_checkpoint_path)

        print("Done")
    else:
        sess.run(tf.global_variables_initializer())  # if the checkpoint doesn't exist, do initialization

    # datareader initialization
    train_dr = dr.DataReader(train_input_path, train_output_path, norm_path, dist_num=config.dist_num, is_training=True, is_shuffle=False)
    valid_path = {'valid_input_path': valid_input_path, 'valid_output_path': valid_output_path, 'norm_path': norm_path}

    for itr in range(config.max_epoch):

        start_time = time.time()
        train_inputs, train_labels = train_dr.next_batch(config.batch_size)
        feed_dict = {m_train.inputs: train_inputs, m_train.labels: train_labels, m_train.keep_prob: config.keep_prob}
        sess.run(m_train.train_op, feed_dict=feed_dict)
        elapsed_time = time.time() - start_time

        # print("time_per_step:%.4f" % elapsed_time)

        if itr % 5 == 0 and itr >= 0:

            train_cost, train_lr = sess.run([m_train.cost, m_train.lr], feed_dict=feed_dict)

            # print("Step: %d, train_cost: %.4f, train_accuracy=%4.4f, train_time=%.4f"
            #       % (itr, train_cost, train_accuracy * 100, el_tim))
            print("Step: %d, train_cost: %.4f, learning_rate: %.7f" % (itr, train_cost, train_lr))
            train_cost_summary_str = sess.run(cost_summary_op, feed_dict={summary_ph: train_cost})
            train_summary_writer.add_summary(train_cost_summary_str, itr)  # write the train phase summary to event files

        if itr % config.val_step == 0 and itr > 0:

            saver.save(sess, logs_dir + "/model.ckpt", itr)  # model save
            print('validation start!')
            valid_cost = do_validation(m_valid, sess, valid_path)

            print("valid_cost: %.4f" % valid_cost)
            valid_cost_summary_str = sess.run(cost_summary_op, feed_dict={summary_ph: valid_cost})
            valid_summary_writer.add_summary(valid_cost_summary_str,
                                             itr)  # write the train phase summary to event files


if __name__ == "__main__":
    # tf.app.run()
    main()
