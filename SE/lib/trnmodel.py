import tensorflow as tf
import utils
import config
import matplotlib


class Model(object):

    def __init__(self, global_step=None, is_training=True):
        self.lr = 0
        self._is_training = is_training
        # self.keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
        if config.mode == 'lstm':

            self.inputs = inputs = tf.placeholder(tf.float32, shape=[None, config.freq_size, 1, 1], name="inputs")
            self.labels = labels = tf.placeholder(tf.float32, shape=[None, config.freq_size], name="labels")

            inputs = tf.squeeze(inputs)

            if is_training:

                inputs_mod = tf.mod(tf.shape(inputs)[0], tf.constant(config.time_width))
                inputs = tf.cond(inputs_mod > 0, lambda: inputs[:-inputs_mod, :], lambda: inputs)
                inputs = tf.reshape(inputs, (-1, config.time_width, config.freq_size))  # time_width == num_steps
                labels = tf.cond(inputs_mod > 0, lambda: labels[:-inputs_mod, :], lambda: labels)
            else:
                inputs = tf.reshape(inputs, (1, -1, config.freq_size))  # time_width == num_steps

        else:
            self.inputs = inputs = tf.placeholder(tf.float32, shape=[None, config.time_width, config.freq_size, 1],
                                                  name="inputs")
            self.labels = labels = tf.placeholder(tf.float32, shape=[None, config.freq_size], name="labels")

        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        logits = self.inference(inputs)
        self.pred = tf.identity(logits, name="pred")
        self.cost = cost = tf.identity((tf.nn.l2_loss(labels - logits))/config.batch_size, name='cost')
        self.raw_cost = tf.reduce_sum((labels - logits)**2, axis=1) / 2

        # self.cost = cost = tf.identity(tf.reduce_mean(tf.losses.huber_loss(labels, logits)), name='cost')

        if is_training:
            trainable_var = tf.trainable_variables()
            self.train_op = self.train(cost, trainable_var, global_step)

    def inference(self, inputs):
        if config.mode is "fcn":
            fm = utils.conv_with_bn(inputs, out_channels=12, filter_size=[config.time_width, 13],
                                    stride=1, act='relu', is_training=self._is_training,
                                    padding="SAME", name="conv_1")

            fm = utils.conv_with_bn(fm, out_channels=16, filter_size=[config.time_width, 11],
                                    stride=1, act='relu', is_training=self._is_training,
                                    padding="SAME", name="conv_2")

            fm = utils.conv_with_bn(fm, out_channels=20, filter_size=[config.time_width, 9],
                                    stride=1, act='relu', is_training=self._is_training,
                                    padding="SAME", name="conv_3")

            fm_skip = utils.conv_with_bn(fm, out_channels=24, filter_size=[config.time_width, 7],
                                    stride=1, act='relu', is_training=self._is_training,
                                    padding="SAME", name="conv_4")

            fm = utils.conv_with_bn(fm_skip, out_channels=32, filter_size=[config.time_width, 7],
                                    stride=1, act='relu', is_training=self._is_training,
                                    padding="SAME", name="conv_5")

            fm = utils.conv_with_bn(fm, out_channels=24, filter_size=[config.time_width, 7],
                                    stride=1, act='relu', is_training=self._is_training,
                                    padding="SAME", name="conv_6") + fm_skip

            fm = utils.conv_with_bn(fm, out_channels=20, filter_size=[config.time_width, 9],
                                    stride=1, act='relu', is_training=self._is_training,
                                    padding="SAME", name="conv_7")

            fm = utils.conv_with_bn(fm, out_channels=16, filter_size=[config.time_width, 11],
                                    stride=1, act='relu', is_training=self._is_training,
                                    padding="SAME", name="conv_8")

            fm = utils.conv_with_bn(fm, out_channels=12, filter_size=[config.time_width, 13],
                                    stride=1, act='relu', is_training=self._is_training,
                                    padding="SAME", name="conv_9")

            fm = utils.conv_with_bn(fm, out_channels=1, filter_size=[config.time_width, config.freq_size],
                                    stride=1, act='linear', is_training=self._is_training,
                                    padding="SAME", name="conv_10")  # (batch_size, 1, config.freq_size, 1)

            # fm = utils.conv_with_bn(fm, out_channels=1, filter_size=[config.time_width, 1],
            #                         stride=1, act='linear', is_training=self._is_training,
            #                         padding="VALID", name="conv_last")
            fm = tf.squeeze(fm, [1, 3])

        elif config.mode is "fnn":

            keep_prob = self.keep_prob

            # inputs = tf.reshape(tf.squeeze(inputs, [3]), (-1, int(config.time_width*config.freq_size)))
            # inputs = tf.nn.dropout(inputs, keep_prob=keep_prob)
            #
            # h1 = tf.nn.relu(utils.batch_norm_affine_transform(inputs, 2048, name='hidden_1',
            #                                                         is_training=self._is_training))
            # # h1 = tf.nn.dropout(h1, keep_prob=keep_prob)
            #
            # h2 = tf.nn.relu(utils.batch_norm_affine_transform(h1, 2048, name='hidden_2',
            #                                                         is_training=self._is_training))
            # # h2 = tf.nn.dropout(h2, keep_prob=keep_prob)
            #
            # # h3 = tf.nn.relu(utils.batch_norm_affine_transform(h2, 2048, name='hidden_3',
            # #                                                         is_training=self._is_training))
            # # h3 = tf.nn.dropout(h3, keep_prob=keep_prob)
            #
            # fm = utils.affine_transform(h2, config.freq_size, name='logits')

            inputs = tf.reshape(tf.squeeze(inputs, [3]), (-1, int(config.time_width*config.freq_size)))
            inputs = tf.nn.dropout(inputs, keep_prob=keep_prob)

            h1 = tf.nn.selu(utils.affine_transform(inputs, 2048, name='hidden_1'))
            h1 = tf.nn.dropout(h1, keep_prob=keep_prob)

            h2 = tf.nn.selu(utils.affine_transform(h1, 2048, name='hidden_2'))
            h2 = tf.nn.dropout(h2, keep_prob=keep_prob)

            h3 = tf.nn.selu(utils.affine_transform(h2, 2048, name='hidden_3'))
            h3 = tf.nn.dropout(h3, keep_prob=keep_prob)

            fm = utils.affine_transform(h3, config.freq_size, name='logits')

        elif config.mode is "sfnn":

            keep_prob = self.keep_prob
            skip_inputs = tf.squeeze(inputs[:, int(config.time_width/2), :])
            inputs = tf.reshape(tf.squeeze(inputs, [3]), (-1, int(config.time_width*config.freq_size)))
            inputs = tf.nn.dropout(inputs, keep_prob=keep_prob)

            h1 = tf.nn.selu(utils.affine_transform(inputs, 2048, name='hidden_1'))
            h1 = tf.nn.dropout(h1, keep_prob=keep_prob)

            h2 = tf.nn.selu(utils.affine_transform(h1, 2048, name='hidden_2'))
            h2 = tf.nn.dropout(h2, keep_prob=keep_prob)

            h3 = tf.nn.selu(utils.affine_transform(h2, 2048, name='hidden_3'))
            h3 = tf.nn.dropout(h3, keep_prob=keep_prob)

            fm = utils.affine_transform(h3, config.freq_size, name='logits')
            fm = fm + skip_inputs

        elif config.mode is "lstm":

            keep_prob = self.keep_prob

            # inputs = tf.squeeze(inputs)[:, int(config.time_width/2), :]

            # inputs = tf.reshape(inputs, (-1, config.time_width, config.freq_size))  # time_width == num_steps
            # inputs = tf.nn.dropout(inputs, keep_prob=keep_prob)

            num_units = [1024, 1024]
            cells = [tf.nn.rnn_cell.LSTMCell(num_units=n, state_is_tuple=True) for n in num_units]

            cell = tf.nn.rnn_cell.MultiRNNCell(cells=cells, state_is_tuple=True)
            cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size=config.freq_size)
            outputs, _state = tf.nn.dynamic_rnn(cell, inputs, time_major=False, dtype=tf.float32)
            fm = tf.reshape(outputs,[-1, config.freq_size])

        return fm

    def train(self, loss, var_list, global_step):

        lrDecayRate = config.lrDecayRate  # 0.99
        lrDecayFreq = config.lrDecayFreq
        momentumValue = .9

        # global_step = tf.Variable(0, trainable=False)
        self.lr = lr = tf.train.exponential_decay(config.lr, global_step, lrDecayFreq, lrDecayRate, staircase=True)

        # define the optimizer
        # optimizer = tf.train.MomentumOptimizer(lr, momentumValue)
        # optimizer = tf.train.AdagradOptimizer(lr)
        #
        optimizer = tf.train.AdamOptimizer(lr)
        grads = optimizer.compute_gradients(loss, var_list=var_list)

        return optimizer.apply_gradients(grads, global_step=global_step)


if __name__ == '__main__':

    global_step = tf.Variable(0, trainable=False)

    Model(is_training=True, global_step=global_step)
