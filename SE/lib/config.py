import numpy as np
import os

time_width = int(9)

fs = float(8000)
win_size = int(0.025 * fs)  # The number of samples in window
win_step = int(0.010 * fs)
# nfft = np.int(2 ** (np.floor(np.log2(win_size) + 1)))
nfft = np.int(256)

freq_size = int(nfft/2+1)

lr = 0.0001
lrDecayRate = .99  # 0.99
lrDecayFreq = 2000

keep_prob = 0.9
global_std = 1.18

device = '/gpu:1'

# logs_dir = os.path.abspath('../logs')

dist_num = int(4)

max_epoch = int(1e6)

batch_size = int(256)

test_batch_size = 128

val_step = int(20)
summary_step = int(500)  # 3000
summary_fnum = int(5)

mode = "fnn"
parallel = False
