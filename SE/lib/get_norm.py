import sys
sys.path.insert(0, './lib')
import os
import numpy as np
import glob
import utils
import scipy.io.wavfile, scipy.signal, scipy.io
import librosa
from multiprocessing import Process, Queue
import config
import matplotlib.pyplot as plt
import math
import numpy as np


def get_stft_mean(input_file_list, output):

    item = []

    power_mean_list = []
    power_gmean_list = []

    for fname in input_file_list:

        data = utils.read_raw(fname)

        # data, rate = librosa.load(fname, config.fs)

        lpsd, _ = utils.get_powerphase(data, config.win_size, config.win_step, config.nfft)

        power_mean = np.mean(lpsd, axis=1)
        power_gmean = np.mean(lpsd)

        power_mean_list.append(power_mean)
        power_gmean_list.append(power_gmean)

    power_mean = np.mean(np.asarray(power_mean_list), axis=0)
    power_gmean = np.mean(np.asarray(power_gmean_list))

    item.append(power_mean)
    item.append(power_gmean)

    output.put(item)


def fnctot_mean(input_file_list, dist_num):

    split_file_list = utils.chunkIt(input_file_list, dist_num)

    queue_list = []
    procs = []

    for i, file_list in enumerate(split_file_list):
        queue_list.append(Queue())  # define queues for saving the outputs of functions
        procs.append(Process(target=get_stft_mean, args=(
            file_list, queue_list[i])))  # define process

    for p in procs:  # process start
        p.start()

    M_list = []

    for i in range(dist_num):  # save results from queues and close queues
        M_list.append(queue_list[i].get())
        queue_list[i].close()

    for p in procs:  # close process
        p.terminate()

    return M_list


def get_stft_std(input_file_list, mean_dic, output):

    item = []

    power_std_list = []
    power_gstd_list = []

    for fname in input_file_list:

        data = utils.read_raw(fname)

        # data, rate = librosa.load(fname, config.fs)

        lpsd, _ = utils.get_powerphase(data, config.win_size, config.win_step, config.nfft)

        power_std = np.mean((lpsd - np.expand_dims(mean_dic['power_mean'], axis=1))**2, axis=1)
        power_gstd = (lpsd - mean_dic['power_gmean'])**2

        power_std_list.append(power_std)
        power_gstd_list.append(power_gstd)

    power_std = np.mean(np.asarray(power_std_list), axis=0)
    power_gstd = np.mean(np.concatenate(power_gstd_list, axis=1))

    item.append(power_std)
    item.append(power_gstd)

    output.put(item)


def fnctot_std(input_file_list, dist_num, mean_dic):

    split_file_list = utils.chunkIt(input_file_list, dist_num)

    queue_list = []
    procs = []

    for i, file_list in enumerate(split_file_list):
        queue_list.append(Queue())  # define queues for saving the outputs of functions
        procs.append(Process(target=get_stft_std, args=(
            file_list, mean_dic, queue_list[i])))  # define process

    for p in procs:  # process start
        p.start()

    M_list = []

    for i in range(dist_num):  # save results from queues and close queues
        M_list.append(queue_list[i].get())
        queue_list[i].close()

    for p in procs:  # close process
        p.terminate()

    return M_list


if __name__ == '__main__':

    # calc_norm()
    distribution_num = 8

    # train_path = os.path.abspath('../data/train/noisy')
    # input_file_list = sorted(glob.glob(train_path + '/*.raw'))
    # result_mean = fnctot_mean(input_file_list, distribution_num)
    # result_mean = np.mean(np.asarray(result_mean), axis=0)
    # result_mean = {'power_mean': result_mean[0], 'power_gmean': result_mean[1]}
    #
    # result_std = fnctot_std(input_file_list, distribution_num, result_mean)
    # result_std = np.mean(np.asarray(result_std), axis=0)
    # result_std = {'power_std': np.sqrt(result_std[0]), 'power_gstd': np.sqrt(result_std[1])}
    #
    # scipy.io.savemat('../data/train/norm/norm_noisy.mat', {**result_mean, **result_std})
    #
    # print('Noisy done')
    #
    # train_path = os.path.abspath('../data/train/clean')
    #
    # input_file_list = sorted(glob.glob(train_path + '/*.raw'))
    # result_mean = fnctot_mean(input_file_list, distribution_num)
    # result_mean = np.mean(np.asarray(result_mean), axis=0)
    # result_mean = {'power_mean': result_mean[0], 'power_gmean': result_mean[1]}
    #
    # result_std = fnctot_std(input_file_list, distribution_num, result_mean)
    # result_std = np.mean(np.asarray(result_std), axis=0)
    # result_std = {'power_std': np.sqrt(result_std[0]), 'power_gstd': np.sqrt(result_std[1])}
    #
    # scipy.io.savemat('../data/train/norm/norm_clean.mat', {**result_mean, **result_std})
    #
    # print('Clean done')

    train_path = os.path.abspath('../enhanced_wav_valid')

    input_file_list = sorted(glob.glob(train_path + '/*.raw'))
    result_mean = fnctot_mean(input_file_list, distribution_num)
    result_mean = np.mean(np.asarray(result_mean), axis=0)
    result_mean = {'power_mean': result_mean[0], 'power_gmean': result_mean[1]}

    result_std = fnctot_std(input_file_list, distribution_num, result_mean)
    result_std = np.mean(np.asarray(result_std), axis=0)
    result_std = {'power_std': np.sqrt(result_std[0]), 'power_gstd': np.sqrt(result_std[1])}

    scipy.io.savemat('../data/train/norm/norm_enhanced_valid.mat', {**result_mean, **result_std})

    print('Enhanced done')
