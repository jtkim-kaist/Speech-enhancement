clc
clear
rng(0)  % seed initialization

system('rm -rf ../SE/data/test/noisy/*');
system('rm -rf ../SE/data/test/clean/*');

timit_list = dirPlus('./speech/timit_coretest', 'FileFilter', '\.(wav|WAV)$');
noise_list = dirPlus('./noise/NOISEX-92_16000');

desired_fs = 8000;

snr_list = [-5, 0, 5];

parpool(4);

%% make noisy

% parfor k = 1:1:length(timit_list)
parfor k = 1:1:10
    k
    for i=1:1:length(noise_list)
        for j=1:1:length(snr_list)
            [audio, speech_fs] = audioread(timit_list{k});
            [noise, noise_fs] = audioread(noise_list{i});
            audio = resample(audio, desired_fs, speech_fs);
            noise = resample(noise, desired_fs, noise_fs);
            noisy_speech = v_addnoise(audio, desired_fs, snr_list(j), 'k', noise, desired_fs);
            noisy_speech = int16(noisy_speech * 32767);            
            fname = sprintf('../SE/data/test/noisy/noisy%02d_snr%02d_num%04d', i, j, k);
            fid_file = fopen([fname, '.raw'], 'w');
            fwrite(fid_file, noisy_speech, 'int16');
            fclose(fid_file);
        end
    end
end
    

%% make clean
parfor i = 1:1:10
% parfor i = 1:1:length(timit_list)
    
    [audio, speech_fs] = audioread(timit_list{i});
    audio = resample(audio, desired_fs, speech_fs);
%     fname = sprintf('./SE_DB/test_wav/clean_num%03d.wav', i);
%     audiowrite(fname, audio, desired_fs);
    audio = int16(audio * 32767);
    fname = sprintf('../SE/data/test/clean/clean_num%04d', i);
    fid_file = fopen([fname, '.raw'], 'w');
    fwrite(fid_file, audio, 'int16');
    fclose(fid_file);
end


% function [] = binary_saver( name_file, data )
% 
%     fid_file = fopen([name_file, '.bin'], 'w');
%     txt_name = [name_file, '_spec'];
%     fid_txt = fopen([txt_name, '.txt'], 'wt');
%     
%     data_size = size(data);
%     
%     fprintf(fid_txt, '%d,%d,float32', data_size(1), data_size(2));
%     fwrite(fid_file, data, 'float32');
%     fclose(fid_file);
%     fclose(fid_txt);
% 
% end

delete(gcp)
