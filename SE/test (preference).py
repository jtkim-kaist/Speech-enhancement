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


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

clean_dir = '/home/jtkim/github/SE_data_test/data/test/clean'
noisy_dir = '/home/jtkim/SE_research/SE_pr_test'
norm_dir = '/home/jtkim/github/SE_data_test//data/train/norm'
recon_save_dir = './test_result_pr'

# clean_dir = os.path.abspath('./data/test/clean')
# noisy_dir = os.path.abspath('./data/test/noisy')
# norm_dir = os.path.abspath('./data/train/norm')


class SE(object):

    def __init__(self, graph_name, norm_path, save_dir = os.path.abspath('./enhanced_wav')):

        graph = gt.load_graph(graph_name)

        self.node_inputs = graph.get_tensor_by_name('prefix/model_1/inputs:0')
        self.node_labels = graph.get_tensor_by_name('prefix/model_1/labels:0')
        if config.mode != 'lstm' and config.mode != 'fcn':
            self.node_keep_prob = graph.get_tensor_by_name('prefix/model_1/keep_prob:0')
        self.node_prediction = graph.get_tensor_by_name('prefix/model_1/pred:0')

        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess_config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=sess_config, graph=graph)
        self.norm_path = norm_path
        self.save_dir = save_dir

    def enhance(self, wav_dir):

        noisy_speech = utils.read_raw(wav_dir)
        temp_dir = './temp/temp.npy'
        np.save(temp_dir, noisy_speech)

        test_dr = dr.DataReader(temp_dir, '', self.norm_path, dist_num=config.dist_num, is_training=False, is_shuffle=False)
        mean, std = test_dr.norm_process(self.norm_path + '/norm_noisy.mat')

        while True:
            test_inputs, test_labels, test_inphase, test_outphase = test_dr.whole_batch(test_dr.num_samples)
            if config.mode != 'lstm' and config.mode != 'fcn':
                feed_dict = {self.node_inputs: test_inputs, self.node_labels: test_labels, self.node_keep_prob: 1.0}
            else:
                feed_dict = {self.node_inputs: test_inputs, self.node_labels: test_labels}

            pred = self.sess.run(self.node_prediction, feed_dict=feed_dict)

            if test_dr.file_change_checker():
                print(wav_dir)

                lpsd = np.expand_dims(np.reshape(pred, [-1, config.freq_size]), axis=2)

                lpsd = np.squeeze((lpsd * std * config.global_std) + mean)

                recon_speech = utils.get_recon(np.transpose(lpsd, (1, 0)), np.transpose(test_inphase, (1, 0)),
                                               win_size=config.win_size, win_step=config.win_step, fs=config.fs)

                test_dr.reader_initialize()

                break

        # file_dir = self.save_dir + '/' + os.path.basename(wav_dir).replace('noisy', 'enhanced').replace('raw', 'wav')
        # librosa.output.write_wav(file_dir, recon_speech, int(config.fs), norm=True)

        return recon_speech


def test(clean_dir=clean_dir, noisy_dir=noisy_dir, norm_dir=norm_dir):

    # logs_dir = os.path.abspath('./logs' + '/logs_' + "2018-06-04-02-06-49")
    model_dir = os.path.abspath('./saved_model')
    # gs.freeze_graph(logs_dir, model_dir, 'model_1/pred,model_1/labels,model_1/cost')

    graph_name = sorted(glob.glob(model_dir + '/*.pb'))[-1]
    # graph_name = '/home/jtkim/hdd3/github_2/SE_graph/Boost_2/Boost_2.pb'

    noisy_list = sorted(glob.glob(noisy_dir + '/*.raw'))
    clean_list = sorted(glob.glob(clean_dir + '/*.raw'))
    num_data = len(clean_list)

    noisy_result = {'noisy_pesq':np.zeros((num_data, 4, 15)),
                     'noisy_stoi':np.zeros((num_data, 4, 15)),
                     'noisy_ssnr':np.zeros((num_data, 4, 15)),
                     'noisy_lsd':np.zeros((num_data, 4, 15))}

    enhance_result = {'enhanced_pesq':np.zeros((num_data, 4, 15)),
                     'enhanced_stoi':np.zeros((num_data, 4, 15)),
                     'enhanced_ssnr':np.zeros((num_data, 4, 15)),
                     'enhanced_lsd':np.zeros((num_data, 4, 15))}

    se = SE(graph_name=graph_name, norm_path=norm_dir)

    eng = matlab.engine.start_matlab()
    eng.addpath(eng.genpath('.'))

    for noisy_dir in noisy_list:

        file_num = int(os.path.basename(noisy_dir).split("_")[-1].split(".raw")[0].split("num")[-1]) - 1
        snr_num = int(os.path.basename(noisy_dir).split("_")[1].split("snr")[1]) - 1
        noise_num = int(os.path.basename(noisy_dir).split("_")[0].split("noisy")[1]) - 1

        for clean_name in clean_list:
            if clean_name.split('num')[-1] == noisy_dir.split('num')[-1]:
                clean_dir = clean_name
                break
        print(noisy_dir)

        # recon_speech = speech_enhance(noisy_dir, graph_name)
        recon_speech = se.enhance(noisy_dir)
        noisy_speech = utils.identity_trans(utils.read_raw(noisy_dir))
        clean_speech = utils.identity_trans(utils.read_raw(clean_dir))

        noisy_measure = utils.se_eval(clean_speech, noisy_speech, float(config.fs), eng)
        enhanced_measure = utils.se_eval(clean_speech, recon_speech, float(config.fs), eng)

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

    eng.exit()


if __name__ == '__main__':

    # clean_dir = os.path.abspath('./data/test/clean')
    # noisy_dir = os.path.abspath('./data/test/noisy')
    # norm_dir = os.path.abspath('./data/train/norm')

    # logs_dir = os.path.abspath('./logs' + '/logs_' + "2018-06-04-02-06-49")
    model_dir = os.path.abspath('./saved_model')
    # gs.freeze_graph(logs_dir, model_dir, 'model_1/pred,model_1/labels,model_1/cost')

    graph_name = sorted(glob.glob(model_dir + '/*.pb'))[-1]
    # graph_name = '/home/jtkim/hdd3/github_2/SE_graph/Boost/Boost.pb'

    noisy_list = sorted(glob.glob(noisy_dir + '/*.raw'))
    num_data = len(noisy_list)

    se = SE(graph_name=graph_name, norm_path=norm_dir)

    computation_time = []
    for noisy_dir in noisy_list:
        fname = recon_save_dir + '/' + os.path.basename(noisy_dir).replace('.raw', '.wav')
        print(noisy_dir)

        # recon_speech = speech_enhance(noisy_dir, graph_name)
        start_time = time.time()
        recon_speech = se.enhance(noisy_dir)
        computation_time.append((time.time() - start_time) / (recon_speech.shape[0]/config.fs) * 1000)
        librosa.output.write_wav(fname, recon_speech, int(config.fs), norm=True)

    print(np.mean(np.asarray(computation_time)))

        # noisy_speech = utils.identity_trans(utils.read_raw(noisy_dir))
