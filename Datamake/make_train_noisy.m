clc
clear
rng(0)  % seed initialization

system('rm -rf ../SE/data/train/noisy/*');
system('rm -rf ../SE/data/train/clean/*');

timit_list = dirPlus('./speech/TIMIT/TRAIN', 'FileFilter', '\.(wav|WAV)$');

hu_list = dirPlus('./noise/Nonspeech', 'FileFilter', '\.(wav|WAV)$');
ad_list = dirPlus('./noise/noise-15', 'FileFilter', '\.(wav|WAV)$');

noise_list = [hu_list; ad_list;];

desired_fs = 8000;

snr_list = [-5, 0, 5, 10, 15, 20];

parpool(4);

aug = 1;

noisy_data = cell(aug, length(timit_list));
clean_data = cell(aug, length(timit_list));
temp_timit_list = timit_list;

for k = 1:1:aug
    k
    timit_indx = randperm(length(temp_timit_list));
    temp_timit_list = temp_timit_list(timit_indx);
    
    parfor i = 1:1:size(temp_timit_list, 1)
        
        snr_indx = randi([1, length(snr_list)]);
        noise_indx = randi([1, length(noise_list)]);
        
        [speech, speech_fs] = audioread(temp_timit_list{i});
        [noise, noise_fs] = audioread(noise_list{noise_indx});
        
        speech = resample(speech, desired_fs, speech_fs);
        noise = resample(noise, desired_fs, noise_fs);
        
        noisy_speech = v_addnoise(speech, desired_fs, snr_list(snr_indx), 'k', noise, noise_fs);
        clean_data{k, i} = speech;
        noisy_data{k, i} = noisy_speech;
        
    end
end

for k = 1:1:aug
    noisy_temp = noisy_data(k, :);
    noisy_temp = cell2mat(noisy_temp');
    noisy_temp = int16(noisy_temp*32767);
    fname = sprintf('../SE/data/train/noisy/noisy_num%03d', k);
    fid_file = fopen([fname, '.raw'], 'w');
    fwrite(fid_file, noisy_temp, 'int16');
    fclose(fid_file);
    
    clean_temp = clean_data(k, :);
    clean_temp = cell2mat(clean_temp');
    clean_temp = int16(clean_temp*32767);
    fname = sprintf('../SE/data/train/clean/clean_num%03d', k);
    fid_file = fopen([fname, '.raw'], 'w');
    fwrite(fid_file, clean_temp, 'int16');
    fclose(fid_file); 
end

delete(gcp);

    