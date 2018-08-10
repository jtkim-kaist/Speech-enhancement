import matlab.engine
import librosa
import tensorflow as tf
import trnmodel
import datareader as dr
import config
import glob
import os
import numpy as np
import time
import utils
from tensorboardX import SummaryWriter
from matplotlib import cm
from timeit import default_timer
from contextlib import contextmanager


class Summary(object):

    def __init__(self, valid_path, logs_dir, name='train'):

        if name == 'train':
            noisy_dir = sorted(glob.glob(valid_path["train_input_path"] + '/*.raw'))
            noisy_dir = noisy_dir[np.random.randint(len(noisy_dir))]
            clean_list = sorted(glob.glob(valid_path["train_output_path"] + '/*.raw'))
        else:
            noisy_dir = sorted(glob.glob(valid_path["valid_input_path"] + '/*.raw'))[config.summary_fnum]
            clean_list = sorted(glob.glob(valid_path["valid_output_path"] + '/*.raw'))

        for clean_name in clean_list:
            if clean_name.split('num')[-1] == noisy_dir.split('num')[-1]:
                clean_dir = clean_name
                break

        self.noisy_dir = noisy_dir
        self.valid_path = valid_path

        self.clean_speech = utils.read_raw(clean_dir)
        self.noisy_speech = utils.read_raw(noisy_dir)

        self.noisy_measure = {}
        if self.clean_speech.shape[0] > 30000:

            self.clean_speech = self.clean_speech[0:30000]
            self.noisy_speech = self.noisy_speech[0:30000]

        self.temp_dir = './data/' + name + '/temp/temp.npy'
        self.name = name
        self.logs_dir = logs_dir
        np.save(self.temp_dir, self.noisy_speech)

    def do_summary(self, m_summary, sess, eng, writer, itr):

        valid_path = self.valid_path
        clean_speech = self.clean_speech
        clean_speech = utils.identity_trans(clean_speech)

        noisy_speech = self.noisy_speech
        noisy_speech = utils.identity_trans(noisy_speech)

        temp_dir = self.temp_dir
        name = self.name
        logs_dir = self.logs_dir

        summary_dr = dr.DataReader(temp_dir, '', valid_path["norm_path"], dist_num=config.dist_num, is_training=False,
                                   is_shuffle=False)

        while True:

            summary_inputs, summary_labels, summary_inphase, summary_outphase = summary_dr.whole_batch(summary_dr.num_samples)

            feed_dict = {m_summary.inputs: summary_inputs, m_summary.labels: summary_labels, m_summary.keep_prob: 1.0}

            pred = sess.run(m_summary.pred, feed_dict=feed_dict)

            if summary_dr.file_change_checker():

                lpsd = np.expand_dims(
                    np.reshape(pred, [-1, config.freq_size]), axis=2)

                mean, std = summary_dr.norm_process(valid_path["norm_path"] + '/norm_noisy.mat')

                lpsd = np.squeeze((lpsd * std * config.global_std) + mean)  # denorm

                recon_speech = utils.get_recon(np.transpose(lpsd, (1, 0)), np.transpose(summary_inphase, (1, 0)),
                                               win_size=config.win_size, win_step=config.win_step, fs=config.fs)

                # plt.plot(recon_speech)
                # plt.show()
                # lab = np.reshape(np.asarray(lab), [-1, 1])
                summary_dr.reader_initialize()
                break

        # write summary

        if itr == config.summary_step:

            self.noisy_measure = utils.se_eval(clean_speech,
                                          np.squeeze(noisy_speech), float(config.fs), eng)
            summary_fname = tf.summary.text(name + '_filename', tf.convert_to_tensor(self.noisy_dir))

            if name == 'train':

                config_str = "<br>sampling frequency: %d</br>" \
                             "<br>window step: %d ms</br>" \
                             "<br>window size: %d ms</br>" \
                             "<br>fft size: %d</br>" \
                             "<br>learning rate: %f</br><br>learning rate decay: %.4f</br><br>learning" \
                             " rate decay frequency: %.4d</br>" \
                             "<br>dropout rate: %.4f</br><br>max epoch:" \
                             " %.4e</br><br>batch size: %d</br><br>model type: %s</br>"\
                             % (config.fs, (config.win_step/config.fs*1000), (config.win_size/config.fs*1000),
                                config.nfft, config.lr, config.lrDecayRate, config.lrDecayFreq, config.keep_prob,
                                config.max_epoch, config.batch_size, config.mode)

                summary_config = tf.summary.text(name + '_configuration', tf.convert_to_tensor(config_str))

                code_list = []
                read_flag = False

                with open('./lib/trnmodel.py', 'r') as f:
                    while True:
                        line = f.readline()
                        if "def inference(self, inputs):" in line:
                            read_flag = True

                        if "return fm" in line:
                            code_list.append('<br>' + line.replace('\n', '') + '</br>')
                            break

                        if read_flag:
                            code_list.append('<br>' + line.replace('\n', '') + '</br>')

                code_list = "<pre>" + "".join(code_list) + "</pre>"

                summary_model = tf.summary.text('train_model', tf.convert_to_tensor(code_list))

                summary_op = tf.summary.merge([summary_fname, summary_config, summary_model])
            else:
                summary_op = tf.summary.merge([summary_fname])

            sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            sess_config.gpu_options.allow_growth = True

            with tf.Session(config=sess_config) as sess:
                summary_writer = tf.summary.FileWriter(logs_dir + '/summary/text')
                text = sess.run(summary_op)
                summary_writer.add_summary(text, 1)

            summary_writer.close()

            writer.add_audio(name + '_audio_ref' + '/clean', clean_speech
                             /np.max(np.abs(clean_speech)), itr,
                             sample_rate=config.fs)
            writer.add_audio(name + '_audio_ref' + '/noisy', noisy_speech
                             /np.max(np.abs(noisy_speech)), itr,
                             sample_rate=config.fs)
            clean_S = get_spectrogram(clean_speech)
            noisy_S = get_spectrogram(noisy_speech)

            writer.add_image(name + '_spectrogram_ref' + '/clean', clean_S, itr)  # image_shape = (C, H, W)
            writer.add_image(name + '_spectrogram_ref' + '/noisy', noisy_S, itr)  # image_shape = (C, H, W)

        enhanced_measure = utils.se_eval(clean_speech, recon_speech, float(config.fs), eng)
        writer.add_scalars(name + '_speech_quality' + '/pesq', {'enhanced': enhanced_measure['pesq'],
                                                                'ref': self.noisy_measure['pesq']}, itr)
        writer.add_scalars(name + '_speech_quality' + '/stoi', {'enhanced': enhanced_measure['stoi'],
                                                                'ref': self.noisy_measure['stoi']}, itr)
        writer.add_scalars(name + '_speech_quality' + '/lsd', {'enhanced': enhanced_measure['lsd'],
                                                               'ref': self.noisy_measure['lsd']}, itr)
        writer.add_scalars(name + '_speech_quality' + '/ssnr', {'enhanced': enhanced_measure['ssnr'],
                                                                'ref': self.noisy_measure['ssnr']}, itr)

        writer.add_audio(name + '_audio_enhanced' + '/enhanced', recon_speech/np.max(np.abs(recon_speech)),
                         itr, sample_rate=config.fs)
        enhanced_S = get_spectrogram(recon_speech)
        writer.add_image(name + '_spectrogram_enhanced' + '/enhanced', enhanced_S, itr)  # image_shape = (C, H, W)


@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start


def get_spectrogram(speech):

    S = librosa.amplitude_to_db(librosa.core.magphase(librosa.stft(speech, hop_length=config.win_step,
                                             win_length=config.win_size, n_fft=config.nfft))[0], ref=np.max)
    S = prepare_spec_image(S)
    return S


def prepare_spec_image(spectrogram):
    spectrogram = (spectrogram - np.min(spectrogram)) / ((np.max(spectrogram)) - np.min(spectrogram))
    spectrogram = np.flip(spectrogram, axis=0)
    return np.uint8(cm.magma(spectrogram) * 255)


def do_validation(m_valid, sess, valid_path):

    # dataset reader setting #

    valid_dr = dr.DataReader(valid_path["valid_input_path"], valid_path["valid_output_path"],
                             valid_path["norm_path"], dist_num=config.dist_num, is_shuffle=False, is_val=True)

    valid_cost_list = []

    while True:

        valid_inputs, valid_labels, valid_inphase, valid_outphase = valid_dr.whole_batch(valid_dr.num_samples)

        feed_dict = {m_valid.inputs: valid_inputs, m_valid.labels: valid_labels,
                     m_valid.keep_prob: 1.0}

        valid_cost = sess.run(m_valid.cost, feed_dict=feed_dict)
        valid_cost_list.append(np.expand_dims(valid_cost, axis=1))

        if valid_dr.file_change_checker():

            valid_dr.file_change_initialize()
            if valid_dr.eof_checker():
                valid_dr.reader_initialize()
                print('Valid data reader was initialized!')  # initialize eof flag & num_file & start index
                break

    valid_cost_list = np.concatenate(valid_cost_list, axis=0)

    total_avg_valid_cost = np.asscalar(np.mean(valid_cost_list))

    return total_avg_valid_cost


def main(argv=None):

    assert not (argv is None), "The project path must be provided."

    # set train path

    train_input_path = argv[0] + '/data/train/noisy'
    train_output_path = argv[0] + '/data/train/clean'
    norm_path = argv[0] + '/data/train/norm'

    # train_input_path = '/home/jtkim/hdd3/github/SE_data_raw/data/train/noisy'
    # train_output_path = '/home/jtkim/hdd3/github/SE_data_raw/data/train/clean'
    # norm_path = '/home/jtkim/hdd3/github/SE_data_raw/data/train/norm'

    # set valid path
    valid_input_path = argv[0] + '/data/valid/noisy'
    valid_output_path = argv[0] + '/data/valid/clean'

    # valid_input_path = '/home/jtkim/hdd3/github/SE_data_raw/data/valid/noisy'
    # valid_output_path = '/home/jtkim/hdd3/github/SE_data_raw/data/valid/clean'

    logs_dir = argv[1]
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

    writer = SummaryWriter(log_dir=logs_dir + '/summary')
    eng = matlab.engine.start_matlab()
    eng.addpath(eng.genpath('.'))
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

    if ckpt and ckpt.model_checkpoint_path:  # model restore

        print("Model restored...")

        saver.restore(sess, ckpt.model_checkpoint_path)

        print("Done")
    else:
        sess.run(tf.global_variables_initializer())  # if the checkpoint doesn't exist, do initialization

    # datareader initialization
    train_dr = dr.DataReader(train_input_path, train_output_path, norm_path, dist_num=config.dist_num, is_training=True, is_shuffle=False)

    valid_path = {'valid_input_path': valid_input_path, 'valid_output_path': valid_output_path, 'norm_path': norm_path}
    train_path = {'train_input_path': train_input_path, 'train_output_path': train_output_path, 'norm_path': norm_path}

    for itr in range(config.max_epoch):

        start_time = time.time()
        train_inputs, train_labels, train_inphase, train_outphase = train_dr.next_batch(config.batch_size)
        feed_dict = {m_train.inputs: train_inputs, m_train.labels: train_labels, m_train.keep_prob: config.keep_prob}
        sess.run(m_train.train_op, feed_dict=feed_dict)
        elapsed_time = time.time() - start_time

        # print("time_per_step:%.4f" % elapsed_time)

        if itr % 5 == 0 and itr >= 0:

            train_cost, train_lr = sess.run([m_train.cost, m_train.lr], feed_dict=feed_dict)

            # print("Step: %d, train_cost: %.4f, train_accuracy=%4.4f, train_time=%.4f"
            #       % (itr, train_cost, train_accuracy * 100, el_tim))
            print("Step: %d, train_cost: %.4f, learning_rate: %.7f" % (itr, train_cost, train_lr))

            writer.add_scalars('training_procedure', {'train': train_cost}, itr)

        if itr % config.val_step == 0 and itr > 0:

            saver.save(sess, logs_dir + "/model.ckpt", itr)  # model save
            print('validation start!')
            valid_cost = do_validation(m_valid, sess, valid_path)

            print("valid_cost: %.4f" % valid_cost)

            writer.add_scalars('training_procedure', {'train': train_cost, 'valid': valid_cost}, itr)

        if itr % config.summary_step == 0 and itr > 0:
            if itr == config.summary_step:
                train_summary = Summary(train_path, logs_dir, name='train')
                valid_summary = Summary(valid_path, logs_dir, name='valid')

                train_summary.do_summary(m_valid, sess, eng, writer, itr)
                valid_summary.do_summary(m_valid, sess, eng, writer, itr)

            else:

                train_summary.do_summary(m_valid, sess, eng, writer, itr)
                valid_summary.do_summary(m_valid, sess, eng, writer, itr)

    writer.close()
    eng.exit()


if __name__ == "__main__":
    # tf.app.run()
    main()
