# Speech Enhancement Toolkit

This toolkit is the implemention of following paper:

J. Kim and M. Hahn, "Speech Enhancement Using a Two-Stage Network for an Efficient Boosting Strategy," in IEEE Signal Processing Letters. doi: 10.1109/LSP.2019.2905660

The speech enhancement (SE) removes the noise signal from the noisy speech signal.

Now, the SE in this toolkit is based on the deep neural network (DNN). And the proposed model will be uploaded.

We hope that this toolkit will contribute as the baselines for SE research area.

This toolkit provides as follows:

- The data generator script for building the noisy training and test dataset from the speech and noise dataset. (MATLAB)

- The training and test script. (python3)

## Prerequisites

- Python3
- [Tensorflow 1.7](https://www.tensorflow.org/)
- [TensorboardX](https://github.com/lanpa/tensorboard-pytorch/tree/master/tensorboardX)
- [librosa](https://librosa.github.io/librosa/)
- [Matlab 2017b](https://kr.mathworks.com/downloads/web_downloads/latest_release)

## Setup

1. Install aformentioned prerequistes.

2. Open the MATLAB and add the directories `./SE` and `./Datamake` including their sub-directories.

3. Install [matlab.engine](https://kr.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)
```
cd "matlabroot/extern/engines/python"
python3 setup.py install
```

## Gererate the training and test data

1. Prepare the speech and noise data. In general, the [TIMIT corpus](https://github.com/philipperemy/timit) is used for the speech data. And, the noise data can be found in [Hu's corpus](http://staff.ustc.edu.cn/~jundu/The%20team/yongxu/demo/115noises.html), [USTC's corpus](https://pan.baidu.com/s/1dER6UUt) and [NOISEX-92](http://www.speech.cs.cmu.edu/comp.speech/Section1/Data/noisex.html).

2. `project_directory(prj_dir)/Datamake/make_train_noisy.m` will make the training set from your data. This code sequentially load the clean speech and synthesize the noisy speech with randomly selected SNR. Here, the type of noise is randomly selected from your training noise dataset. To reduce the file number, this code concatenate all generated noisy speech. Therefore, if your RAM is not enough, you should modify the code. All generated data will be written in '.raw' format with 'int16' datatype.

3. `project_directory(prj_dir)/Datamake/make_test_noisy.m` will make the test set from your data. This code sequentially load the clean speech and synthesize the noisy speech with desired SNR. Here, the code use all types of noises in the test noise dataset when synthesize the noisy speech. All generated data will be written in '.raw' format with 'int16' datatype.

#### Usage of `make_train_noisy.m`

Before run the code, move your training speech and noise dataset by referring the below code.

```
% prj_dir/Datamake/make_train_noisy.m
timit_list = dirPlus('./speech/TIMIT/TRAIN', 'FileFilter', '\.(wav|WAV)$');

hu_list = dirPlus('./noise/Nonspeech', 'FileFilter', '\.(wav|WAV)$');
ad_list = dirPlus('./noise/noise-15', 'FileFilter', '\.(wav|WAV)$');
```

#### Options

- You can set the SNRs for noisy speech by adjusting `snr_list`. 
- You can make more fluent data by adjusiting `aug`.

#### Results

The generated dataset will be saved in `prj_dir/SE/data/train/noisy` and `prj_dir/SE/data/train/clean`.

#### Usage of `make_test_noisy.m`

Before run the code, move your test speech and noise dataset by referring the below code.

```
% prj_dir/Datamake/make_test_noisy.m
timit_list = dirPlus('./speech/timit_coretest', 'FileFilter', '\.(wav|WAV)$');
noise_list = dirPlus('./noise/NOISEX-92_16000');
```

#### Options

- You can set the SNRs for noisy speech by adjusting `snr_list`. 

#### Results

The generated dataset will be saved in `prj_dir/SE/data/test/noisy` and `prj_dir/SE/data/test/clean`.

#### Validation dataset

To run the code, the validation set is needed. I used to randomly select about 50 noisy utterances with corresponding clean utterances from test set then, move these to `prj_dir/SE/data/valid/noisy` and `prj_dir/SE/data/valid/clean`.

## Gererate the normalize factor

This code conduct Z-score normalization to the input features, so that some normalization factor from training dataset is needed.

To get the normalization factor, just run the `prj_dir/SE/get_norm.py`

#### Options

- You can use the multiple core by adjusting `distribution_num`.

#### Results

The generated normalization factor will be saved in `prj_dir/SE/data/train/norm`.

## Training

Just run the `prj_dir/SE/main.py`

#### Model

You can check the training model in `prj_dir/SE/lib/trnmodel.py` 

#### Configuration

You can check the training configuration in `prj_dir/SE/lib/config.py`

## Tensorboard

While training, you can use the tensorboard for monitoring the training procedure.

`tensorboard --logdir='prj_dir/SE/logs_dir/your log directory'`

This toolkit supports followings:

- PESQ, STOI, LSD, SSNR (Objective measure).

![alt tag](https://user-images.githubusercontent.com/24668469/40900963-0fdeb11e-6809-11e8-806b-1aaa98620632.PNG)

- Clean, noisy, and enhanced spectrogram.

![alt tag](https://user-images.githubusercontent.com/24668469/40900992-29dd83ec-6809-11e8-8991-9255f429de12.PNG)

- Clean, noisy and enhanced wavs.

![alt tag](https://user-images.githubusercontent.com/24668469/40900998-30d1f1f6-6809-11e8-8250-54d22f9fcee7.PNG)

- Configuration

![alt tag](https://user-images.githubusercontent.com/24668469/40901003-358057c4-6809-11e8-9b8b-a1b13848bc97.PNG)

## Reference

[1] Xu, Yong, et al. "A regression approach to speech enhancement based on deep neural networks." IEEE/ACM Transactions on Audio, Speech and Language Processing (TASLP) 23.1 (2015): 7-19.

[2] Brookes, Mike. (2011). Voicebox: Speech Processing Toolbox for Matlab. 

[3] Jacob, SoundZone_Tools, (2017), GitHub repository, https://github.com/JacobD10/SoundZone_Tools

[4] Loizou, P.C.: "Speech enhancement: theory and practice", (CRC press, 2013), pp. 83−84

