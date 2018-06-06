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

from time import sleep


def speech_enhance(wav_dir, graph_name):

    noisy_speech = utils.read_raw(wav_dir)

    temp_dir = './temp/temp.npy'
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

    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess_config.gpu_options.allow_growth = True

    while True:

        test_inputs, test_labels = test_dr.next_batch(config.test_batch_size)

        feed_dict = {node_inputs: test_inputs, node_labels: test_labels, node_keep_prob: 1.0}

        with tf.Session(graph=graph, config=sess_config) as sess:
            pred_temp, lab_temp = sess.run([node_prediction, node_labels], feed_dict=feed_dict)

        pred.append(pred_temp)
        lab.append(lab_temp)

        # print(test_dr.file_change_checker())
        if test_dr.file_change_checker():
            print(wav_dir)
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

    clean_dir = os.path.abspath('./data/test/clean')
    noisy_dir = os.path.abspath('./data/test/noisy')
    logs_dir = os.path.abspath('./logs' + '/logs_' + "2018-06-04-02-06-49")
    model_dir = os.path.abspath('./enhanced_wav_valid/enhanced_model')
    save_dir = os.path.abspath('./enhanced_wav_valid')

    gs.freeze_graph(logs_dir, model_dir, 'model_1/pred,model_1/labels,model_1/cost')

    noisy_list = sorted(glob.glob(noisy_dir + '/*.raw'))
    clean_list = sorted(glob.glob(clean_dir + '/*.raw'))

    graph_name = sorted(glob.glob(model_dir + '/*.pb'))[-1]

    norm_path = os.path.abspath('./data/train/norm')

    noisy_result = {'noisy_pesq':np.zeros((10, 3, 15)),
                     'noisy_stoi':np.zeros((10, 3, 15)),
                     'noisy_ssnr':np.zeros((10, 3, 15)),
                     'noisy_lsd':np.zeros((10, 3, 15))}

    enhance_result = {'enhanced_pesq':np.zeros((10, 3, 15)),
                     'enhanced_stoi':np.zeros((10, 3, 15)),
                     'enhanced_ssnr':np.zeros((10, 3, 15)),
                     'enhanced_lsd':np.zeros((10, 3, 15))}

    for noisy_dir in noisy_list:

        file_num = int(os.path.basename(noisy_dir).split("_")[-1].split(".raw")[0].split("num")[-1]) - 1
        snr_num = int(os.path.basename(noisy_dir).split("_")[1].split("snr")[1]) - 1
        noise_num = int(os.path.basename(noisy_dir).split("_")[0].split("noisy")[1]) - 1

        for clean_name in clean_list:
            if clean_name.split('num')[-1] == noisy_dir.split('num')[-1]:
                clean_dir = clean_name
                break
        print(noisy_dir)

        recon_speech = speech_enhance(noisy_dir, graph_name)
        noisy_speech = utils.identity_trans(utils.read_raw(noisy_dir))
        clean_speech = utils.identity_trans(utils.read_raw(clean_dir))

        noisy_measure = utils.se_eval(clean_speech, noisy_speech, float(config.fs))
        enhanced_measure = utils.se_eval(clean_speech, recon_speech, float(config.fs))

        noisy_result['noisy_pesq'][file_num, snr_num, noise_num] = noisy_measure['pesq']
        noisy_result['noisy_stoi'][file_num, snr_num, noise_num] = noisy_measure['stoi']
        noisy_result['noisy_ssnr'][file_num, snr_num, noise_num] = noisy_measure['ssnr']
        noisy_result['noisy_lsd'][file_num, snr_num, noise_num] = noisy_measure['lsd']

        enhance_result['enhanced_pesq'][file_num, snr_num, noise_num] = enhanced_measure['pesq']
        enhance_result['enhanced_stoi'][file_num, snr_num, noise_num] = enhanced_measure['stoi']
        enhance_result['enhanced_ssnr'][file_num, snr_num, noise_num] = enhanced_measure['ssnr']
        enhance_result['enhanced_lsd'][file_num, snr_num, noise_num] = enhanced_measure['lsd']

    noisy_result['noisy_pesq'] = np.mean(noisy_result['noisy_pesq'], axis=0)
    noisy_result['noisy_stoi'] = np.mean(noisy_result['noisy_stoi'], axis=0)
    noisy_result['noisy_ssnr'] = np.mean(noisy_result['noisy_ssnr'], axis=0)
    noisy_result['noisy_lsd'] = np.mean(noisy_result['noisy_lsd'], axis=0)

    enhance_result['enhanced_pesq'] = np.mean(enhance_result['enhanced_pesq'], axis=0)
    enhance_result['enhanced_stoi'] = np.mean(enhance_result['enhanced_stoi'], axis=0)
    enhance_result['enhanced_ssnr'] = np.mean(enhance_result['enhanced_ssnr'], axis=0)
    enhance_result['enhanced_lsd'] = np.mean(enhance_result['enhanced_lsd'], axis=0)

    scipy.io.savemat('./test_result/noisy_result.mat', noisy_result)
    scipy.io.savemat('./test_result/enhanced_result.mat', enhance_result)

    # file_dir = save_dir + '/' + os.path.basename(wav_dir).replace('noisy', 'enhanced')
        # utils.write_bin(recon_speech, np.max(np.abs(recon_speech)), file_dir)
