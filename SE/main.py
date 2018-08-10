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


if __name__ == '__main__':

    reset = True
    test_only = False
    train_reset = True

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

        # os.popen('rm -rf ' + '/home/jtkim/hdd3/github/SE_data_raw/data/train/noisy/*.npy')
        # os.popen('rm -rf ' + '/home/jtkim/hdd3/github/SE_data_raw/data/train/noisy/*.bin')
        #
        # os.popen('rm -rf ' + '/home/jtkim/hdd3/github/SE_data_raw/data/train/clean/*.npy')
        # os.popen('rm -rf ' + '/home/jtkim/hdd3/github/SE_data_raw/data/train/clean/*.bin')
        #
        # os.popen('rm -rf ' + '/home/jtkim/hdd3/github/SE_data_raw/data/valid/noisy/*.npy')
        # os.popen('rm -rf ' + '/home/jtkim/hdd3/github/SE_data_raw/data/valid/noisy/*.bin')
        #
        # os.popen('rm -rf ' + '/home/jtkim/hdd3/github/SE_data_raw/data/valid/clean/*.npy')
        # os.popen('rm -rf ' + '/home/jtkim/hdd3/github/SE_data_raw/data/valid/clean/*.bin')

    # model train

    if not test_only:

        tr.main([prj_dir, logs_dir])

        # save graph

        gs.freeze_graph(logs_dir, save_dir, 'model_1/pred,model_1/labels,model_1/cost')

    print("Training was ended!")
