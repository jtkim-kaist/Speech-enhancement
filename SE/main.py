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


def se_test(wav_dir, noise_dir, snr, noise_type=1):

    # clean_speech, clean_fs = librosa.load(wav_dir, config.fs)
    clean_speech = utils.read_raw(wav_dir)
    eng = matlab.engine.start_matlab()

    # noisy_speech = np.array(eng.noise_add(wav_dir, noise_dir, noise_type, snr, nargout=1))
    # noisy_speech, noisy_fs = librosa.load(noise_dir, config.fs)
    noisy_speech = utils.read_raw(noise_dir)

    # noisy_measure = se_eval(clean_speech, np.squeeze(noisy_speech), float(config.fs))

    temp_dir = './data/test/temp/temp.npy'

    np.save(temp_dir, noisy_speech)
    graph_name = sorted(glob.glob('./saved_model/*.pb'))[-1]
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

    # os.popen('rm -rf ' + temp_dir)

    noisy_measure = utils.se_eval(clean_speech[0:recon_speech.shape[0]], np.squeeze(noisy_speech[0:recon_speech.shape[0]]), float(config.fs))

    enhanced_measure = utils.se_eval(clean_speech[0:recon_speech.shape[0]], recon_speech, float(config.fs))
    print("pesq: %.4f -> %.4f" % (noisy_measure["pesq"], enhanced_measure["pesq"]))
    print("lsd: %.4f -> %.4f" % (noisy_measure["lsd"], enhanced_measure["lsd"]))
    print("stoi: %.4f -> %.4f" % (noisy_measure["stoi"], enhanced_measure["stoi"]))
    print("ssnr: %.4f -> %.4f" % (noisy_measure["ssnr"], enhanced_measure["ssnr"]))

    plt.subplot(3, 1, 1)
    S = librosa.amplitude_to_db(librosa.stft(clean_speech[0:recon_speech.shape[0]], hop_length=config.win_step,
                                             win_length=config.win_size, n_fft=config.nfft), ref=np.max)
    ld.specshow(S, y_axis='linear', hop_length=config.win_step, sr=config.fs)

    plt.subplot(3, 1, 2)
    S = librosa.amplitude_to_db(librosa.stft(np.squeeze(noisy_speech[0:recon_speech.shape[0]]), hop_length=config.win_step,
                                             win_length=config.win_size, n_fft=config.nfft), ref=np.max)
    ld.specshow(S, y_axis='linear', hop_length=config.win_step, sr=config.fs)

    plt.subplot(3, 1, 3)
    S = librosa.amplitude_to_db(librosa.stft(recon_speech, hop_length=config.win_step,
                                             win_length=config.win_size, n_fft=config.nfft), ref=np.max)
    ld.specshow(S, y_axis='linear', hop_length=config.win_step, sr=config.fs)

    plt.show()

    return recon_speech


if __name__ == '__main__':

    reset = False
    test_only = False
    train_reset = False

    if reset:
        os.popen('rm -rf ./logs/*')

    print("The size of log directory: " + utils.du('./logs'))

    if "G" in utils.du('./logs'):

        assert (float(utils.du('./logs').split("G")[0]) < 5.), "The size of log directory is more than > 5G, please refresh" \
                                                        " this directory"

    logs_dir = os.path.abspath('./logs' + '/logs_' + time.strftime("%Y-%m-%d-%H-%M-%S"))

    os.popen('mkdir ' + logs_dir)
    os.popen('mkdir ' + logs_dir + '/summary')

    save_dir = os.path.abspath('./saved_model')
    prj_dir = os.path.abspath('.')

    os.popen('rm -rf ./saved_model/*')

    if train_reset:
        os.popen('rm -rf ' + './data/train/noisy/*.npy')
        os.popen('rm -rf ' + './data/train/noisy/*.bin')

        os.popen('rm -rf ' + './data/train/clean/*.npy')
        os.popen('rm -rf ' + './data/train/clean/*.bin')

        os.popen('rm -rf ' + './data/valid/noisy/*.npy')
        os.popen('rm -rf ' + './data/valid/noisy/*.bin')

        os.popen('rm -rf ' + './data/valid/clean/*.npy')
        os.popen('rm -rf ' + './data/valid/clean/*.bin')

    # model train

    if not test_only:

        tr.main([prj_dir, logs_dir])

        # save graph

        gs.freeze_graph(logs_dir, save_dir, 'model_1/pred,model_1/labels,model_1/cost')


    # test graph

    # test_noise_dir = './data/test/noise/NOISEX-92_16000'
    # wav_dir = os.path.abspath('./data/test/clean/clean_num005.wav')
    # test_noise_dir = './data/test/noise/noise-15'
    # wav_dir = os.path.abspath('./train_clean_num001.wav')

    test_noise_dir = '/home/jtkim/hdd3/github/SE_datamake/datamake/SE_DB2/test/noisy/noisy15_snr02_num0010.raw'
    # test_noise_dir = './data/valid/noisy/noisy15_snr03_num0010.raw'
    wav_dir = os.path.abspath('./data/valid/clean/clean_num0010.raw')
    #
    recon_dir = os.path.abspath('./enhanced_wav/' + os.path.basename(wav_dir).replace("clean", "enhanced"))
    snr = 5.0
    recon_speech = se_test(wav_dir, test_noise_dir, snr, noise_type=15)
    # librosa.output.write_wav(recon_dir, recon_speech, config.fs, norm=True)
