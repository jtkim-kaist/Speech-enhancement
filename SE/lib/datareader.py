import numpy as np
import os, sys
import glob
import utils
import librosa
import time
import config
import math
import random
from multiprocessing import Process, Queue
import scipy.io, scipy.io.wavfile, scipy.signal
from sklearn.feature_extraction import image
import matplotlib.pyplot as plt
import librosa.display as ld


class DataReader(object):

    def __init__(self, input_dir, output_dir, norm_dir, dist_num, is_training=True, is_shuffle=True, is_val=False,
                 is_summary=False):
        # is_training: True (training, validation)
        # is_training: False (test)

        self._is_shuffle = is_shuffle
        self._dataindx = 0
        self._perm_indx = 0

        self._is_training = is_training
        self._win_size = config.win_size
        self._time_width = config.time_width
        self._dist_num = dist_num
        self._inpur_dir = input_dir
        self._output_dir = output_dir
        self._norm_dir = norm_dir
        self._batch_size = 0
        if self._is_training:
            self._input_file_list = sorted(glob.glob(input_dir+'/*.raw'))
        else:
            self._input_file_list = [input_dir]  # wave directory
        if is_summary:
            self._input_file_list = [input_dir]  # wave directory
        self.phase = []  # phase for reconstruction
        if is_training:
            self._output_file_list = sorted(glob.glob(output_dir+'/*.raw'))

        self._file_len = len(self._input_file_list)

        self._num_file = 0
        self._start_idx = 0

        self._is_val = is_val
        self._is_summary = is_summary
        self.lb = False
        self.eof = False
        self.file_change = False
        self.num_samples = 0

        self._inputs, self._input_phase = self._read_input(
            self._input_file_list[self._num_file])  # (batch_size, time, freq, 1)

        if self._is_training:
            self._outputs, self._output_phase = self._read_output()  # (batch_size, freq)

            assert np.shape(self._inputs)[0] == np.shape(self._outputs)[0],\
                ("# samples is not matched between input: %d and output: %d files"
                 % (np.shape(self._inputs)[0], np.shape(self._outputs)[0]))

            data_num = list(set([fname.split("num")[1].split('.raw')[0] for fname in self._input_file_list]))
            data_num.sort()

            assert len(data_num) == len(self._output_file_list)

        # self._train_mean, self._train_std = self.norm_process(norm_dir+'/norm.mat')

    @staticmethod
    def norm_process(norm_dir):

        norm_param = scipy.io.loadmat(norm_dir)

        power_mean = norm_param["power_mean"]
        power_std = norm_param["power_std"]

        train_mean = np.transpose(power_mean, (1, 0))
        train_std = np.transpose(power_std, (1, 0))

        return train_mean, train_std

    def next_batch(self, batch_size):

        self._batch_size = batch_size

        if self._start_idx + batch_size > self.num_samples:

            inputs = self._inputs[self._start_idx:, :]
            input_phase = self._input_phase[self._start_idx:, :]

            if self._is_training:
                outputs = self._outputs[self._start_idx:, :]
                output_phase = self._output_phase[self._start_idx:, :]
            else:
                outputs = np.zeros((inputs.shape[0], inputs.shape[2]))
                output_phase = np.zeros((inputs.shape[0], inputs.shape[2]))

            self._start_idx = 0
            self.file_change = True
            self._num_file += 1

            if self._num_file > self._file_len - 1:  # final file check
                self.eof = True

                if self._is_val:
                    pass
                else:
                    self._num_file = 0
                    self._inputs, self._input_phase = self._read_input(
                        self._input_file_list[self._num_file])  # (batch_size, time, freq, 1)
                    if self._is_training:
                        self._outputs, self._output_phase = self._read_output()
            else:
                self.eof = False
                self._inputs, self._input_phase = self._read_input(self._input_file_list[self._num_file])  # (batch_size, time, freq, 1)
                if self._is_training:
                    self._outputs, self._output_phase = self._read_output()

        else:
            self.file_change = False

            inputs = self._inputs[self._start_idx:self._start_idx + batch_size, :]
            input_phase = self._input_phase[self._start_idx:self._start_idx + batch_size, :]

            if self._is_training:
                outputs = self._outputs[self._start_idx:self._start_idx + batch_size, :]
                output_phase = self._output_phase[self._start_idx:self._start_idx + batch_size, :]
            else:
                outputs = np.zeros((inputs.shape[0], inputs.shape[2]))
                output_phase = np.zeros((inputs.shape[0], inputs.shape[2]))

            self._start_idx += batch_size

        return inputs, outputs, input_phase, output_phase

    def whole_batch(self, batch_size):

        self._batch_size = batch_size

        inputs = self._inputs
        input_phase = self._input_phase

        if self._is_training:
            outputs = self._outputs
            output_phase = self._output_phase
        else:
            outputs = np.zeros((inputs.shape[0], inputs.shape[2]))
            output_phase = np.zeros((inputs.shape[0], inputs.shape[2]))

        self.file_change = True
        self._num_file += 1

        if self._num_file > self._file_len - 1:  # final file check
            self.eof = True

            if self._is_val:
                pass
            else:
                self._num_file = 0
                self._inputs, self._input_phase = self._read_input(
                    self._input_file_list[self._num_file])  # (batch_size, time, freq, 1)
                if self._is_training:
                    self._outputs, self._output_phase = self._read_output()
        else:
            self.eof = False
            self._inputs, self._input_phase = self._read_input(self._input_file_list[self._num_file])  # (batch_size, time, freq, 1)
            if self._is_training:
                self._outputs, self._output_phase = self._read_output()

        return inputs, outputs, input_phase, output_phase

    @staticmethod
    def _normalize(mean, std, x):
        x = (x - mean)/std
        return x

    def _read_input(self, input_file_dir):

        # print("num_file: %.2d" % self._num_file)
        # self._dataindx = input_file_dir.split("num")[1].split('.')[0]
        # dataname = os.path.dirname(input_file_dir) + '/' + os.path.basename(input_file_dir).split('.')[0] + '.npy'

        if self._is_training:

            # print("num_file: %.3d" % self._num_file)
            self._dataindx = input_file_dir.split("num")[1].split('.')[0]
            dataname = os.path.dirname(input_file_dir) + '/' + os.path.basename(input_file_dir).split('.')[0] + '.bin'

            if os.path.exists(dataname):
                # print("num_file: %.2d load..." % self._num_file)

                feat_spec = np.load(dataname.replace('.bin', '.npy')).item()
                feat_size = feat_spec['feat_size']
                feat_shape = feat_spec['feat_shape']
                phase_shape = feat_spec['phase_shape']
                feat_max = feat_spec['max']
                feat_phase = utils.read_raw(dataname) * np.float32(feat_max)
                feat = feat_phase[0:feat_size].reshape(feat_shape)
                phase = feat_phase[feat_size:].reshape(phase_shape)
                # print("num_file: %.2d load... done" % self._num_file)

            else:
                print("num_file: %.2d feature extraction..." % self._num_file)
                # data, _ = librosa.load(input_file_dir, config.fs)
                data = utils.read_raw(input_file_dir)
                # plt.subplot(211)
                #
                # S = librosa.amplitude_to_db(
                #     librosa.stft(data, hop_length=config.win_step,
                #                  win_length=config.win_size, n_fft=config.nfft), ref=np.max)
                # ld.specshow(S, y_axis='linear', hop_length=config.win_step, sr=config.fs)
                if config.parallel:
                    feat, phase = self.lpsd_dist_p(data, self._dist_num, is_patch=True)
                else:
                    feat, phase = self.lpsd_dist(data, self._dist_num, is_patch=True)
                feat_size = feat.size
                feat_phase = np.concatenate((feat.reshape(-1), phase.reshape(-1)))
                utils.write_bin(feat_phase, np.max(np.abs(feat_phase)), dataname)
                feat_spec = {'feat_size': feat_size, 'phase_shape': phase.shape, 'feat_shape': feat.shape, 'max': np.max(np.abs(feat_phase))}
                np.save(dataname.replace('.bin', ''), feat_spec)

                print("num_file: %.2d feature extraction... done." % self._num_file)

        else:
            data = np.load(input_file_dir)

            # data, _ = librosa.load(input_file_dir, config.fs)
            if config.parallel:
                feat, phase = self.lpsd_dist_p(data, self._dist_num)
            else:
                feat, phase = self.lpsd_dist(data, self._dist_num)

        # feat shape: (num samples, config.time_width, config.freq_size, 1)

        # if self._is_shuffle:
        #     feat = np.reshape(feat, (-1, self._batch_size, feat.shape[1], feat.shape[2], feat.shape[3]))
        #     self._perm_indx = perm_indx = np.random.permutation(feat.shape[0])
        #     feat = np.reshape(feat[perm_indx, :], (-1, config.time_width, config.freq_size, 1))

        ''''''
        if self._is_shuffle:
            self._perm_indx = perm_indx = np.random.permutation(feat.shape[0])
            feat = feat[perm_indx, :]

        self.num_samples = np.shape(feat)[0]

        if self._is_val:
            phase = np.zeros(feat.shape)

        return feat, phase

    def _read_output(self):

        for i, fname in enumerate(self._output_file_list):\

            fname = fname.split('num')[-1].split('.raw')[0]

            if self._dataindx == fname:
                # print(self._output_file_list[i])
                dataname = os.path.dirname(self._output_file_list[i]) + \
                           '/' + os.path.basename(self._output_file_list[i]).split('.')[0] + '.bin'
                if os.path.exists(dataname):
                    feat_spec = np.load(dataname.replace('.bin', '.npy')).item()
                    feat_size = feat_spec['feat_size']
                    feat_shape = feat_spec['feat_shape']
                    phase_shape = feat_spec['phase_shape']
                    feat_max = feat_spec['max']
                    feat_phase = utils.read_raw(dataname) * np.float32(feat_max)
                    feat = feat_phase[0:feat_size].reshape(feat_shape)
                    phase = feat_phase[feat_size:].reshape(phase_shape)

                else:
                    data = utils.read_raw(self._output_file_list[i])
                    if config.parallel:
                        feat = self.lpsd_dist_p(data, self._dist_num,
                                          is_patch=False)  # (The number of samples, config.freq_size, 1, 1)
                    else:
                        feat, phase = self.lpsd_dist(data, self._dist_num,
                                          is_patch=False)  # (The number of samples, config.freq_size, 1, 1)
                    feat_size = feat.size
                    feat_phase = np.concatenate((feat.reshape(-1), phase.reshape(-1)))
                    feat_spec = {'feat_size': feat_size, 'phase_shape': phase.shape, 'feat_shape': feat.shape,
                                 'max': np.max(np.abs(feat_phase))}

                    utils.write_bin(feat_phase, np.max(np.abs(feat_phase)), dataname)
                    np.save(dataname.replace('.bin', ''), feat_spec)
                # plt.subplot(212)
                # S = librosa.amplitude_to_db(
                #     librosa.stft(data, hop_length=config.win_step,
                #                  win_length=config.win_size, n_fft=config.nfft), ref=np.max)
                # ld.specshow(S, y_axis='linear', hop_length=config.win_step, sr=config.fs)
                #
                # plt.show()
                # data, _ = librosa.load(self._output_file_list[i], config.fs)
                break

        # print('output')
        # if self._is_shuffle:
        #
        #     feat = np.reshape(feat, (-1, self._batch_size, feat.shape[1], feat.shape[2], feat.shape[3]))
        #     feat = np.reshape(feat[self._perm_indx, :], (-1, config.freq_size, 1, 1))

        ''''''
        if self._is_shuffle:

            feat = feat[self._perm_indx, :]

        return np.squeeze(feat), phase

    def lpsd_dist(self, data, dist_num, is_patch=True):

        result = self.get_lpsd(data)

        # result = np.asarray(M_list)
        # print(result.shape)
        # result = np.reshape(result, (-1, result.shape[2], result.shape[3]))

        lpsd = np.expand_dims(result[:, :, 0], axis=2)  # expand_dims for normalization (shape matching for broadcast)

        if not self._is_training:
            self.phase.append(result[:, :, 1])

        pad = np.expand_dims(np.zeros((int(config.time_width/2), lpsd.shape[1])), axis=2)  # pad for extracting the patches

        if is_patch:
            mean, std = self.norm_process(self._norm_dir + '/norm_noisy.mat')
            lpsd = self._normalize(mean, std, lpsd)
        else:
            mean, std = self.norm_process(self._norm_dir + '/norm_noisy.mat')
            lpsd = self._normalize(mean, std, lpsd)
        # print(result.shape)

        if is_patch:
            lpsd = np.squeeze(np.concatenate((pad, lpsd, pad), axis=0))  # padding for patching
            # print(result.shape)
            lpsd = image.extract_patches_2d(lpsd, (config.time_width, lpsd.shape[1]))

        lpsd = np.expand_dims(lpsd, axis=3)
        phase = result[:, :, 1]

        return lpsd, phase

    def lpsd_dist_p(self, data, dist_num, is_patch=True):

        data = data.tolist()
        chunk_list = utils.chunkIt(data, dist_num)
        data_list = []
        for item in chunk_list:
            if len(item) > 1:
                data_list.append(item)

        while True:

            queue_list = []
            procs = []

            for queue_val in queue_list:
                print(queue_val.empty())

            for i, data in enumerate(data_list):
                queue_list.append(Queue())  # define queues for saving the outputs of functions

                procs.append(Process(target=self.get_lpsd_p, args=(
                    data, queue_list[i])))  # define process

            for queue_val in queue_list:
                print(queue_val.empty())

            for p in procs:  # process start
                p.start()

            for queue_val in queue_list:
                while queue_val.empty():
                    pass

            for queue_val in queue_list:
                print(queue_val.empty())

            M_list = []

            for i in range(dist_num):  # save results from queues and close queues
                # while not queue_list[i].empty():
                get_time = time.time()
                M_list.append(queue_list[i].get(timeout=3))
                get_time = time.time() - get_time
                queue_list[i].close()
                queue_list[i].join_thread()

            for p in procs:  # close process
                p.terminate()

            if get_time < 3:
                break
            else:
                print('Some error occurred, restarting the lpsd extraction...')

        result = np.concatenate(M_list, axis=0)
        # result = np.asarray(M_list)
        # print(result.shape)
        # result = np.reshape(result, (-1, result.shape[2], result.shape[3]))

        lpsd = np.expand_dims(result[:, :, 0], axis=2)  # expand_dims for normalization (shape matching for broadcast)

        # if not self._is_training:
        self.phase.append(result[:, :, 1])

        pad = np.expand_dims(np.zeros((int(config.time_width/2), lpsd.shape[1])), axis=2)  # pad for extracting the patches

        if is_patch:
            mean, std = self.norm_process(self._norm_dir + '/norm_noisy.mat')
            lpsd = self._normalize(mean, std, lpsd)
        else:
            mean, std = self.norm_process(self._norm_dir + '/norm_noisy.mat')
            lpsd = self._normalize(mean, std, lpsd)
        # print(result.shape)

        if is_patch:
            lpsd = np.squeeze(np.concatenate((pad, lpsd, pad), axis=0))  # padding for patching
            # print(result.shape)
            lpsd = image.extract_patches_2d(lpsd, (config.time_width, lpsd.shape[1]))

        lpsd = np.expand_dims(lpsd, axis=3)

        return lpsd

    @staticmethod
    def get_lpsd_p(data, output):

        data = np.asarray(data).astype(dtype=np.float32)
        # nfft = np.int(2**(np.floor(np.log2(self._nperseg)+1)))

        # _, _, Zxx = scipy.signal.stft(data, fs=fs, nperseg=self._nperseg, nfft=int(nfft))

        lpsd, phase = utils.get_powerphase(data, config.win_size, config.win_step, config.nfft)  # (freq, time)
        lpsd = np.transpose(np.expand_dims(lpsd, axis=2), (1, 0, 2))[:-1, :]
        phase = np.transpose(np.expand_dims(phase, axis=2), (1, 0, 2))[:-1, :]
        result = np.concatenate((lpsd, phase), axis=2)
        print("put start ")

        output.put(result)
        print("put done")
        sys.exit()

    @staticmethod
    def get_lpsd(data):

        data = np.asarray(data).astype(dtype=np.float32)
        # nfft = np.int(2**(np.floor(np.log2(self._nperseg)+1)))

        # _, _, Zxx = scipy.signal.stft(data, fs=fs, nperseg=self._nperseg, nfft=int(nfft))

        lpsd, phase = utils.get_powerphase(data, config.win_size, config.win_step, config.nfft)  # (freq, time)
        lpsd = np.transpose(np.expand_dims(lpsd, axis=2), (1, 0, 2))[:-1, :]
        phase = np.transpose(np.expand_dims(phase, axis=2), (1, 0, 2))[:-1, :]
        result = np.concatenate((lpsd, phase), axis=2)
        return result

    @staticmethod
    def _padding(inputs, batch_size):
        pad_size = batch_size - inputs.shape[0] % (batch_size)
        pad_shape = (pad_size,) + inputs.shape[1:]
        inputs = np.concatenate((inputs, np.zeros(pad_shape, dtype=np.float32)))

        # window_pad = np.zeros((w_val, inputs.shape[1]))
        # inputs = np.concatenate((window_pad, inputs, window_pad), axis=0)
        return inputs

    def reader_initialize(self):
        self._num_file = 0
        self._start_idx = 0
        self.phase = []
        self.eof = False
        self.lb = False

    def eof_checker(self):
        return self.eof

    def file_change_checker(self):
        return self.file_change

    def file_change_initialize(self):
        self.file_change = False

    def is_lastbatch(self):
        return self.lb


# def worker_test(jobs):
#     while True:
#         tmp = jobs.get()
#
#         if tmp==None:
#
#             break
#         else:
#             return tmp


if __name__ == '__main__':

    dist_num = 4
    train_input_path = os.path.abspath('../data/train/noisy')
    train_output_path = os.path.abspath('../data/train/clean')
    norm_path = os.path.abspath('../data/train/norm')

    train_dr = DataReader(train_input_path, train_output_path, norm_path, dist_num, is_training=True)

    while True:
        sample = train_dr.next_batch(512)  # input: (batch_size, time, freq, channel) output: (batch_size, time, freq)
        print("num_file : " + str(train_dr._num_file))
