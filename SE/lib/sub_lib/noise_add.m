function [noisy_speech] = noise_add(wav_dir, noise_dir, noise_type, snr)

    desired_fs = 8000.0;
    
    noise_list = dirPlus(noise_dir, 'FileFilter', '\.(wav)$');
    [audio, speech_fs] = audioread(wav_dir);
    noise_list{noise_type}
    [noise, noise_fs] = audioread(noise_list{noise_type});
    
     audio = resample(audio, desired_fs, speech_fs);
     noise = resample(noise, desired_fs, noise_fs);

    noisy_speech = v_addnoise(audio, desired_fs, snr, 'k', noise, desired_fs);
%        noisy_speech = audio + noise(1:length(audio));
%     subplot(2,1,1)
%     plot(audio)
%     subplot(2,1,2)
%     plot(noisy_speech)
end