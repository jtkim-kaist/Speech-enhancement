import matlab.engine

import sys
import subprocess

sys.path.insert(0, './lib')
import os
import datareader as dr
import train as tr
import graph_save as gs
import graph_test as gt
import glob
import config
import scipy.io
import tensorflow as tf
import numpy as np
import utils
import matplotlib.pyplot as plt
import time
import librosa
import librosa.display as ld
import get_norm as gn


def speech_enhance(wav_dir, graph_name):

    noisy_speech = utils.read_raw(wav_dir)
    temp_dir = './enhanced_wav/temp/temp.npy'
    np.save(temp_dir, noisy_speech)
    graph = gt.load_graph(graph_name)
    norm_path = os.path.abspath('./data/train/norm')

    test_dr = dr.DataReader(temp_dir, '', norm_path, dist_num=config.dist_num, is_training=False, is_shuffle=False)

    node_inputs = graph.get_tensor_by_name('prefix/model_1/inputs:0')
    node_labels = graph.get_tensor_by_name('prefix/model_1/labels:0')
    node_keep_prob = graph.get_tensor_by_name('prefix/model_1/keep_prob:0')
    node_prediction = graph.get_tensor_by_name('prefix/model_1/pred:0')

    pred = []
    lab = []

    while True:

        test_inputs, test_labels = test_dr.next_batch(config.test_batch_size)

        feed_dict = {node_inputs: test_inputs, node_labels: test_labels, node_keep_prob: 1.0}

        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess_config.gpu_options.allow_growth = True

        with tf.Session(graph=graph, config=sess_config) as sess:
            pred_temp, lab_temp = sess.run([node_prediction, node_labels], feed_dict=feed_dict)

        pred.append(pred_temp)
        lab.append(lab_temp)

        if test_dr.file_change_checker():
            phase = test_dr.phase[0]

            lpsd = np.expand_dims(np.reshape(np.concatenate(pred, axis=0), [-1, config.freq_size])[0:phase.shape[0], :], axis=2)

            mean, std = test_dr.norm_process(norm_path + '/norm_noisy.mat')

            lpsd = np.squeeze((lpsd * std) + mean)  # denorm

            recon_speech = utils.get_recon(np.transpose(lpsd, (1, 0)), np.transpose(phase, (1, 0)),
                                           win_size=config.win_size, win_step=config.win_step, fs=config.fs)

            # plt.plot(recon_speech)
            # plt.show()
            # lab = np.reshape(np.asarray(lab), [-1, 1])
            test_dr.reader_initialize()
            break

    return recon_speech


if __name__ == '__main__':

    input_dir = os.path.abspath('./data/train/noisy')
    logs_dir = os.path.abspath('./logs' + '/logs_' + time.strftime("%Y-%m-%d-%H-%M-%S"))
    model_dir = os.path.abspath('./enhanced_wav/enhance_model')
    save_dir = os.path.abspath('./enhanced_wav')
    gs.freeze_graph(logs_dir, model_dir, 'model_1/pred,model_1/labels,model_1/cost')

    input_file_list = sorted(glob.glob(input_dir + '/*.raw'))
    graph_name = sorted(glob.glob(save_dir + '/*.pb'))[-1]
    norm_path = os.path.abspath('./data/train/norm')

    for wav_dir in input_file_list:
        recon_speech = speech_enhance(wav_dir, graph_name)
        file_dir = save_dir + '/' + os.path.basename(wav_dir).replace('noisy', 'enhanced')
        utils.write_bin(recon_speech, np.max(np.abs(recon_speech)), file_dir)
