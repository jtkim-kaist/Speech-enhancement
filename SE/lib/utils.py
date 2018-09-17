import matlab.engine
import tensorflow as tf
import numpy as np
import librosa
import os, sys
import subprocess
import matplotlib.pyplot as plt
import config
from multiprocessing import Process, Queue


def se_eval(clean, noisy, fs, eng):

    clean = matlab.double(clean.tolist())
    noisy = matlab.double(noisy.tolist())
    pesq, stoi, ssnr, lsd = eng.se_eval(clean, noisy, fs, nargout=4)

    measure = {"pesq": pesq, "stoi": stoi, "ssnr": ssnr, "lsd": lsd}

    return measure


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def conv2d_basic(x, W, bias, stride=1, padding="SAME"):
    # conv = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
    conv = tf.nn.conv2d(x, W, strides=[1, 100, stride, 1], padding=padding)  # 100 is for reducing the time dimension

    return tf.nn.bias_add(conv, bias)


def conv_with_bn(inputs, out_channels, filter_size=[3, 3], stride=1, act='relu', is_training=True,
                 padding="SAME", name=None):

    in_height = filter_size[0]
    in_width = filter_size[1]
    in_channels = inputs.get_shape().as_list()[3]
    W = weight_variable([in_height, in_width, in_channels, out_channels], name=name+'_W')
    b = bias_variable([out_channels], name=name+'_b')
    conv = conv2d_basic(inputs, W, b, stride=stride, padding=padding)
    conv = tf.contrib.layers.batch_norm(conv, decay=0.9, is_training=is_training, updates_collections=None)
    if act is 'relu':
        # relu = tf.nn.relu(conv)

        # prelu = tf.keras.layers.PReLU(alpha_initializer='zeros', shared_axes=[1, 2])
        relu = tf.nn.selu(conv)
    elif act is 'linear':
        relu = conv
    return relu


def affine_transform(x, output_dim, name=None):
    """
    affine transformation Wx+b
    assumes x.shape = (batch_size, num_features)
    """

    w = tf.get_variable(name + "_w", [x.get_shape()[1], output_dim],
                        initializer=tf.truncated_normal_initializer(stddev=0.02))
    b = tf.get_variable(name + "_b", [output_dim], initializer=tf.constant_initializer(0.0))

    return tf.matmul(x, w) + b


def weight_variable(shape, stddev=0.02, name=None):

    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def frame2rawlabel(label, win_len, win_step):

    num_frame = label.shape[0]

    total_len = (num_frame-1) * win_step + win_len
    raw_label = np.zeros((total_len, 1))
    start_indx = 0

    i = 0

    while True:

        if start_indx + win_len > total_len:
            break
        else:
            temp_label = label[i]
            raw_label[start_indx+1:start_indx+win_len] = raw_label[start_indx+1:start_indx+win_len] + temp_label
        i += 1

        start_indx = start_indx + win_step

    raw_label = (raw_label >= 1).choose(raw_label, 1)

    return raw_label


def Truelabel2Trueframe( TrueLabel_bin,wsize,wstep ):
    iidx = 0
    Frame_iidx = 0
    Frame_len = Frame_Length(TrueLabel_bin, wstep, wsize)
    Detect = np.zeros([Frame_len, 1])
    while 1 :
        if iidx+wsize <= len(TrueLabel_bin) :
            TrueLabel_frame = TrueLabel_bin[iidx:iidx + wsize - 1]*10
        else:
            TrueLabel_frame = TrueLabel_bin[iidx:]*10

        if (np.sum(TrueLabel_frame) >= wsize / 2) :
            TrueLabel_frame = 1
        else :
            TrueLabel_frame = 0

        if (Frame_iidx >= len(Detect)):
            break

        Detect[Frame_iidx] = TrueLabel_frame
        iidx = iidx + wstep
        Frame_iidx = Frame_iidx + 1
        if (iidx > len(TrueLabel_bin)):
            break

    return Detect


def Frame_Length( x,overlap,nwind ):
    nx = len(x)
    noverlap = nwind - overlap
    framelen = int((nx - noverlap) / (nwind - noverlap))
    return framelen


def get_powerphase(s, win_size, win_step, nfft):

    S = librosa.stft(np.squeeze(s), hop_length=win_step, win_length=win_size, n_fft=nfft)
    mag, phase = librosa.magphase(S)

    phase = np.angle(phase)
    lpsd = librosa.core.amplitude_to_db(mag)

    return lpsd, phase


def get_recon(lpsd, phase, win_size, win_step, fs=16000):
    '''
    :param lpsd: (freq, time)
    :param phase: (freq, time)
    :param win_size:
    :param win_step:
    :param fs:
    :return:
    '''

    mag = librosa.core.db_to_amplitude(lpsd)
    S_recon = mag*(np.cos(phase) + 1j*np.sin(phase))
    s_recon = librosa.istft(S_recon, hop_length=win_step, win_length=win_size)
    return s_recon


def batch_norm_affine_transform(x, output_dim, decay=0.9, name=None, seed=0, is_training=True):
    """
    affine transformation Wx+b
    assumes x.shape = (batch_size, num_features)
    """
    # initializer = tf.contrib.layers.xavier_initializer(seed=seed)

    w = tf.get_variable(name+"_w", [x.get_shape()[1], output_dim], initializer=tf.contrib.layers.xavier_initializer(seed=seed))
    b = tf.get_variable(name+"_b", [output_dim], initializer=tf.constant_initializer(0.0))
    affine_result = tf.matmul(x, w) + b
    batch_norm_result = tf.contrib.layers.batch_norm(affine_result, decay=decay, is_training=is_training,
                                                     updates_collections=None)
    return batch_norm_result


def affine_transform(x, output_dim, seed=0, name=None):
    """
    affine transformation Wx+b
    assumes x.shape = (batch_size, num_features)
    """
    initializer = tf.truncated_normal_initializer(stddev=0.02, seed=seed)

    # weights = tf.get_variable(name + "_w", [x.get_shape()[1], output_dim],
    #                           initializer=tf.contrib.layers.xavier_initializer(seed=seed))
    weights = tf.get_variable(name + "_w", [x.get_shape()[1], output_dim],
                              initializer=initializer)
    b = tf.get_variable(name + "_b", [output_dim], initializer=tf.constant_initializer(0.0))

    return tf.matmul(x, weights) + b


def read_bin(input_file_dir):

    input_spec_dir = os.path.dirname(input_file_dir) +\
                     '/' + os.path.basename(input_file_dir).split('.bin')[0] + '_spec.txt'
    input_file_dir.split('.bin')[0] + '_spec.txt'
    data = np.fromfile(input_file_dir, dtype=np.float32)  # (# total frame, feature_size)
    with open(input_spec_dir,'r') as f:
        spec = f.readline()
        size = spec.split(',')
    data = data.reshape((int(size[0]), int(size[1])), order='F')
    data = np.squeeze(data)

    return data


def read_raw(input_file_dir):

    data = np.fromfile(input_file_dir, dtype=np.int16)  # (# total frame, feature_size)
    data = np.float32(data) / 32767.
    data = np.squeeze(data)

    return data


def write_bin(data, max_val, file_dir):
    data = np.int16(data / max_val * 32767)
    with open(file_dir, 'wb') as f:
        data.tofile(f)


def du(path):
    return subprocess.check_output(['du', '-sh', path]).split()[0].decode('utf-8')


def identity_trans(data):

    data = np.asarray(data).astype(dtype=np.float32)
    if config.parallel:
        lpsd, phase = lpsd_dist_p(data, config.dist_num)
    else:
        lpsd, phase = lpsd_dist(data)

    # lpsd, phase = get_powerphase(data, win_size, win_step, nfft)  # (freq, time)
    # lpsd = np.transpose(np.expand_dims(lpsd, axis=2), (1, 0, 2))[:-1, :]
    # phase = np.transpose(np.expand_dims(phase, axis=2), (1, 0, 2))[:-1, :]
    result = get_recon(lpsd, phase, config.win_size, config.win_step, fs=config.fs)
    return result


def lpsd_dist_p(data, dist_num):

    data = data.tolist()
    data_list = chunkIt(data, dist_num)

    phase = []
    queue_list = []
    procs = []

    for queue_val in queue_list:
        print(queue_val.empty())

    for i, data in enumerate(data_list):
        queue_list.append(Queue())  # define queues for saving the outputs of functions

        procs.append(Process(target=get_lpsd_p, args=(
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
        M_list.append(queue_list[i].get())
        queue_list[i].close()
        queue_list[i].join_thread()

    for p in procs:  # close process
        p.terminate()

    result = np.concatenate(M_list, axis=0)
    # result = np.asarray(M_list)
    # print(result.shape)
    # result = np.reshape(result, (-1, result.shape[2], result.shape[3]))

    lpsd = np.transpose(result[:, :, 0], (1, 0))  # expand_dims for normalization (shape matching for broadcast)
    phase = np.transpose(result[:, :, 1], (1, 0))

    return lpsd, phase


def get_lpsd_p(data, output):

    data = np.asarray(data).astype(dtype=np.float32)
    # nfft = np.int(2**(np.floor(np.log2(self._nperseg)+1)))

    # _, _, Zxx = scipy.signal.stft(data, fs=fs, nperseg=self._nperseg, nfft=int(nfft))

    lpsd, phase = get_powerphase(data, config.win_size, config.win_step, config.nfft)  # (freq, time)
    lpsd = np.transpose(np.expand_dims(lpsd, axis=2), (1, 0, 2))[:-1, :]
    phase = np.transpose(np.expand_dims(phase, axis=2), (1, 0, 2))[:-1, :]
    result = np.concatenate((lpsd, phase), axis=2)
    print("put start ")
    output.put(result)
    print("shit done")
    sys.exit()


def lpsd_dist(data):

    result = get_lpsd(data)
    lpsd = np.transpose(result[:, :, 0], (1, 0))  # expand_dims for normalization (shape matching for broadcast)
    phase = np.transpose(result[:, :, 1], (1, 0))

    return lpsd, phase


def get_lpsd(data):

    data = np.asarray(data).astype(dtype=np.float32)
    # nfft = np.int(2**(np.floor(np.log2(self._nperseg)+1)))

    # _, _, Zxx = scipy.signal.stft(data, fs=fs, nperseg=self._nperseg, nfft=int(nfft))

    lpsd, phase = get_powerphase(data, config.win_size, config.win_step, config.nfft)  # (freq, time)
    lpsd = np.transpose(np.expand_dims(lpsd, axis=2), (1, 0, 2))[:-1, :]
    phase = np.transpose(np.expand_dims(phase, axis=2), (1, 0, 2))[:-1, :]
    result = np.concatenate((lpsd, phase), axis=2)
    return result


def extract_patch(inputs, patch_size):
    # inputs: Tensor, shape=(batch_size, width)
    # patch_size: tuple, shape=(patch_height, patch_width)
    # outputs: Tensor, shape=(# patches (batch_size - patch_height + 1), patch_height, patch_width)

    inputs = tf.expand_dims(tf.expand_dims(inputs, axis=0), axis=3)
    kernel = tf.reshape(tf.diag(tf.ones(patch_size[0], 1)), shape=(patch_size[0], 1, 1, patch_size[0]))
    conv = tf.nn.conv2d(input=inputs, filter=kernel, strides=[1, config.freq_size, 1, 1], padding='VALID', name='patch_conv')
    conv = tf.transpose(tf.squeeze(conv), (0, 2, 1))

    return conv


def conv2d_basic_2(x, W, bias, stride=1, padding="SAME"):
    # conv = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
    conv = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)  # 100 is for reducing the time dimension

    return tf.nn.bias_add(conv, bias)


def conv_with_bn_2(inputs, out_channels, filter_size=[3, 3], stride=1, act='relu', scale=True, is_training=True,
                 padding="SAME", name=None):

    in_height = filter_size[0]
    in_width = filter_size[1]
    in_channels = inputs.get_shape().as_list()[3]
    W = weight_variable([in_height, in_width, in_channels, out_channels], name=name+'_W')
    b = bias_variable([out_channels], name=name+'_b')
    conv = conv2d_basic_2(inputs, W, b, stride=stride, padding=padding)
    # conv = tf.contrib.layers.batch_norm(conv, scale=scale, is_training=is_training)
    if act is 'relu':
        relu = tf.nn.selu(conv)

        # prelu = tf.keras.layers.PReLU(alpha_initializer='zeros', shared_axes=[1, 2])
        # relu = tf.nn.selu(conv)
    elif act is 'linear':
        relu = conv
    return relu