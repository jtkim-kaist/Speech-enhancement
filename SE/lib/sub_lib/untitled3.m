clc
clear
close all
wav_dir = '/home/jtkim/hdd3/github/SE_datamake/datamake/timit_coretest/DR1/FELC0/SA1.WAV';
% wav_dir = '/home/jtkim/hdd3/github/SE/data/test/clean/clean_num022.wav';
[clean, fs] = audioread(wav_dir);

noise_dir = '/home/jtkim/hdd3/github/SE_datamake/datamake/noise/NOISEX-92_16000';

noise_type = 15;
snr = 5;
noisy_speech = noise_add(wav_dir, noise_dir, noise_type, snr);
[zz,enhanced] = WienerNoiseReduction(noisy_speech,16000);

% clean = resample(clean, 8000, fs);

a = snr_mean(clean, noisy_speech);
[pesq, stoi_val, snr, lsd] = se_eval(clean, noisy_speech, fs)

enhanced = enhanced ./ max(abs(enhanced));
[pesq2, stoi_val2, snr2, lsd2] = se_eval(clean(1:length(enhanced)), enhanced, fs)
% noisy_speech = noisy_speech / max(abs(noisy_speech));
% audiowrite('./sample.wav', noisy_speech, 8000)

subplot(3,1,1)
plot(clean)
subplot(3,1,2)
plot(noisy_speech)
subplot(3,1,3)
plot(enhanced)

close all