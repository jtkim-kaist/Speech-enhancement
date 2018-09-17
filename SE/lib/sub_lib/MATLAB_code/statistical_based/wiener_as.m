function wiener_as(filename,outfile)

%
%  Implements the Wiener filtering algorithm based on a priori SNR estimation [1].
% 
%  Usage:  wiener_as(noisyFile, outputFile)
%           
%         infile - noisy speech file in .wav format
%         outputFile - enhanced output file in .wav format

%         
%  Example call:  wiener_as('sp04_babble_sn10.wav','out_wien_as.wav');
%
%  References:
%   [1] Scalart, P. and Filho, J. (1996). Speech enhancement based on a priori 
%       signal to noise estimation. Proc. IEEE Int. Conf. Acoust. , Speech, Signal 
%       Processing, 629-632.
%   
% Authors: Yi Hu and Philipos C. Loizou
%
% Copyright (c) 2006 by Philipos C. Loizou
% $Revision: 0.0 $  $Date: 10/09/2006 $
%-------------------------------------------------------------------------

if nargin<2
   fprintf('Usage: wiener_as(noisyfile.wav,outFile.wav) \n\n');
   return;
end



[noisy_speech, fs, nbits]= wavread( filename);
noisy_speech= noisy_speech; 
% column vector noisy_speech

% set parameter values
mu= 0.98; % smoothing factor in noise spectrum update
a_dd= 0.98; % smoothing factor in priori update
eta= 0.15; % VAD threshold
frame_dur= 20; % frame duration 
L= frame_dur* fs/ 1000; % L is frame length (160 for 8k sampling rate)
hamming_win= hamming( L); % hamming window
U= ( hamming_win'* hamming_win)/ L; % normalization factor

% first 120 ms is noise only
len_120ms= fs/ 1000* 120;
% first_120ms= noisy_speech( 1: len_120ms).* ...
%     (hann( len_120ms, 'periodic'))';
first_120ms= noisy_speech( 1: len_120ms);

% =============now use Welch's method to estimate power spectrum with
% Hamming window and 50% overlap
nsubframes= floor( len_120ms/ (L/ 2))- 1;  % 50% overlap
noise_ps= zeros( L, 1);
n_start= 1; 
for j= 1: nsubframes
    noise= first_120ms( n_start: n_start+ L- 1);
    noise= noise.* hamming_win;
    noise_fft= fft( noise, L);
    noise_ps= noise_ps+ ( abs( noise_fft).^ 2)/ (L* U);
    n_start= n_start+ L/ 2; 
end
noise_ps= noise_ps/ nsubframes;
%==============

% number of noisy speech frames 
len1= L/ 2; % with 50% overlap
nframes= floor( length( noisy_speech)/ len1)- 1; 
n_start= 1; 

for j= 1: nframes
    noisy= noisy_speech( n_start: n_start+ L- 1);
    noisy= noisy.* hamming_win;
    noisy_fft= fft( noisy, L);
    noisy_ps= ( abs( noisy_fft).^ 2)/ (L* U);
    
    % ============ voice activity detection
    if (j== 1) % initialize posteri
        posteri= noisy_ps./ noise_ps;
        posteri_prime= posteri- 1; 
        posteri_prime( find( posteri_prime< 0))= 0;
        priori= a_dd+ (1-a_dd)* posteri_prime;
    else
        posteri= noisy_ps./ noise_ps;
        posteri_prime= posteri- 1;
        posteri_prime( find( posteri_prime< 0))= 0;
        priori= a_dd* (G_prev.^ 2).* posteri_prev+ ...
            (1-a_dd)* posteri_prime;
    end

    log_sigma_k= posteri.* priori./ (1+ priori)- log(1+ priori);    
    vad_decision(j)= sum( log_sigma_k)/ L;    
    if (vad_decision(j)< eta) 
        % noise only frame found
        noise_ps= mu* noise_ps+ (1- mu)* noisy_ps;
        vad( n_start: n_start+ L- 1)= 0;
    else
        vad( n_start: n_start+ L- 1)= 1;
    end
    % ===end of vad===
    
    G= sqrt( priori./ (1+ priori)); % gain function
   
    enhanced= ifft( noisy_fft.* G, L);
        
    if (j== 1)
        enhanced_speech( n_start: n_start+ L/2- 1)= ...
            enhanced( 1: L/2);
    else
        enhanced_speech( n_start: n_start+ L/2- 1)= ...
            overlap+ enhanced( 1: L/2);  
    end
    
    overlap= enhanced( L/ 2+ 1: L);
    n_start= n_start+ L/ 2; 
    
    G_prev= G; 
    posteri_prev= posteri;
    
end

enhanced_speech( n_start: n_start+ L/2- 1)= overlap; 

wavwrite( enhanced_speech, fs, nbits, outfile);

    
