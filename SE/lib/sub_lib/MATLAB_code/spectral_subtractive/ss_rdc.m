function ss_rdc( filename,outfile)
%
%  Implements the spectral subtraction algorithm with reduced-delay 
%  convolution and adaptive averaging [1].
% 
%  Usage:  mband(infile, outputfile,Nband,Freq_spacing)
%           
%         infile - noisy speech file in .wav format
%         outputFile - enhanced output file in .wav format
%         Nband - Number of frequency bands (recommended 4-8)
%         Freq_spacing - Type of frequency spacing for the bands, choices:
%                        'linear', 'log' and 'mel'
%
%  Example call:  ss_rdc('sp04_babble_sn10.wav','out_rdc.wav');
%
%  References:
%   [1] Gustafsson, H., Nordholm, S., and Claesson, I. (2001). Spectral sub-
%       traction using reduced delay convolution and adaptive averaging. IEEE 
%       Trans. on Speech and Audio Processing, 9(8), 799-807.
%   
% Authors: Yi Hu and Philipos C. Loizou
%
% Copyright (c) 2006 by Philipos C. Loizou
% $Revision: 0.0 $  $Date: 10/09/2006 $
%-------------------------------------------------------------------------

if nargin<2
   fprintf('Usage: ss_rdc noisyfile.wav outFile.wav \n\n');
   return;
end

[noisy_speech, fs, nbits]= wavread( filename);
noisy_speech= noisy_speech'; % change to row vector

if fs== 8000
    L= 160; M= 32; N= 256;
elseif fs== 16000
    L= 320; M= 64; N= 512;
else
    exit( 'incorrect sampling rate!\n');
end

% set parameter values
mu= 0.98; % smoothing factor in noise spectrum update
a= 0.98; % smoothing factor in priori update
eta= 0.15; % VAD threshold
gamma_c= 0.8; % smoothing factor in G_M update
beta= 0.7; % oversubtraction factor
k= L/ M; % number of segments of M
hann_win= hamming( L); % hanning window
hann_win= hann_win'; % change to row vector

% first 120 ms is noise only
len_120ms= fs/ 1000* 120;
% first_120ms= noisy_speech( 1: len_120ms).* ...
%     (hann( len_120ms, 'periodic'))';
first_120ms= noisy_speech( 1: len_120ms);
nsubframes= len_120ms/ M;  % L is 20ms

% now use Bartlett method to estimate power spectrum
noise_ps= zeros( 1, M);
for j= 1: nsubframes
    noise= first_120ms( (j- 1)* M+ 1: j* M);
    noise_fft= fft( noise, M);
    noise_ps= noise_ps+ ( abs( noise_fft).^ 2)/ M;
end
noise_ps= noise_ps/ nsubframes;
P_w= sqrt( noise_ps);
% plot( P_w);

% number of noisy speech frames
nframes= floor( length( noisy_speech)/ L); % noisy_speech( nframes* L)= 0;
enhanced_speech= zeros( 1, (nframes- 1)* L+ N);

for j= 1: nframes
    %noisy= noisy_speech( (j-1)* L+ 1: j* L).* hann_win;    
    noisy= noisy_speech( (j-1)* L+ 1: j* L);    
    x_ps= zeros( 1, M);
    for n= 1: k
        x= noisy( (n-1)* M+ 1: n* M);
        x_fft= fft( x, M);
        x_ps= x_ps+ (abs( x_fft).^ 2)/ M;
    end
    x_ps= x_ps/ k; 
    P_x= sqrt( x_ps); % magnitude spectrum for noisy 
    
    % voice activity detection
    if (j== 1) % initialize posteri
        posteri= (P_x.^ 2)./ (P_w.^ 2);
        posteri_prime= posteri- 1; 
        posteri_prime( find( posteri_prime< 0))= 0;
        priori= a+ (1-a)* posteri_prime;
    else
        posteri_old= posteri;
        posteri= (P_x.^ 2)./ (P_w.^ 2);;
        posteri_prime= posteri- 1;
        posteri_prime( find( posteri_prime< 0))= 0;
        priori= a* (G_M2.^ 2).* posteri_old+ (1-a)* posteri_prime;      
    end

    log_sigma_k= posteri.* priori./ (1+ priori)- log(1+ priori);
    
    vad_decision(j)= sum( log_sigma_k)/ M;    
    
    if (vad_decision(j)< eta) % noise only frame found
        P_w= mu* P_w+ (1- mu)* P_x;
        vad( (j-1)*L+1: j*L)= 0;
    else
        vad( (j-1)*L+1: j*L)= 1;
    end
    % ===end of vad===
    
    G_M= 1- beta* (P_w./ P_x); % gain function
    G_M( find( G_M< 0))= 0;
    
    % spectrum discrepancy
    beta_i= min( 1, sum( abs( P_x- P_w))/ sum( P_w));
    alpha_1= 1- beta_i;    
     
    if (j== 1) % initialize alpha_2
        alpha_2= alpha_1; 
    end
    
    if (alpha_2< alpha_1)
        alpha_2= gamma_c* alpha_2+ (1- gamma_c)* alpha_1;
    else
        alpha_2= alpha_1;
    end
    
    if (j== 1)
        G_M2= (1- alpha_2)* G_M;
    else
        G_M2= alpha_2* G_M2+ (1- alpha_2)* G_M;
    end
    
    % impulse response of G_M2   
    G_M2_ir= firls(M, (0: M-1)/M, G_M2);    
    G_M2_intpl= fft( G_M2_ir, N);    
    

    noisy_freq= fft( noisy, N);
    
    enhanced= ifft( noisy_freq.* G_M2_intpl, N);

    if (j== 1)
        enhanced_speech( 1: N)= enhanced;
    else
        overlap= enhanced( 1: N- L)+ enhanced_speech( (j-1)* ...
            L+ 1: (j-1)*L+ N- L);
        enhanced_speech( (j-1)*L+ 1: (j-1)*L+ N- L)= overlap;
        enhanced_speech( (j-1)*L+ N- L+ 1: (j-1)*L+ N)= ...
            enhanced( N- L+ 1: N);
    end    
    
    
end

wavwrite( enhanced_speech, fs, nbits, outfile);


    
