function [pesq, stoi_val, segsnr_mean, lsd] = se_eval(clean, noisy, fs)

%[clean, clean_fs] = audioread(clean_name);
%[noisy, ~] = audioread(enhanced_name);

res = pesq_mex_vec(clean, noisy, fs);
pesq=mos2pesq(res);


stoi_val = stoi(double(clean), double(noisy), fs);

clean     = clean - mean(clean);
noisy = noisy - mean(noisy);
noisy = noisy/max(abs(noisy));
clean = clean/max(abs(clean));

lsd = LogSpectralDistance(clean,noisy,fs);
% clean = clean/max(abs(clean));
% noisy = noisy/max(abs(noisy));
[snr_mean, segsnr_mean] = comp_snr(clean, noisy, fs);
% size(clean')
% size(noisy')
% snr = segsnr(clean', noisy', fs);
% plot(noisy - clean);
clc
end

