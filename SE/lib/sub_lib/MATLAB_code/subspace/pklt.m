function pklt( noisy_file, outfile)
%
%  Implements a perceptually-motivated subspace algorithm  [1].
%  
%
%  Usage:  pklt(noisyFile, outputFile)
%           
%         infile - noisy speech file in .wav format
%         outputFile - enhanced output file in .wav format
%  
%
%  Example call:  pklt('sp04_babble_sn10.wav','out_pklt.wav');
%
%  References:
%   [1] Jabloun, F. and Champagne, B. (2003). Incorporating the human hearing
%   	 properties in the signal subspace approach for speech enhancement. IEEE
%   	 Trans. on Speech and Audio Processing, 11(6), 700-708.
%   
% Authors: Yi Hu and Philipos C. Loizou
%
% Copyright (c) 2006 by Philipos C. Loizou
% $Revision: 0.0 $  $Date: 10/09/2006 $
%-------------------------------------------------------------------------

if nargin<2
   fprintf('Usage: pklt(noisyfile.wav,outFile.wav) \n\n');
   return;
end

vad_thre= 1.2; % 2.378 is the value from Mittal's paper
mu_vad= 0.98; % mu to use in vad

[noisy_speech, Srate, NBITS]= wavread( noisy_file);
subframe_dur= 4;  %subframe length is 4 ms
len= floor( Srate* subframe_dur/ 1000);    
P= len; % sub-frame length - for 8k sampling rate, len is 32
frame_dur= 32; % frame length in msecs
N= frame_dur* Srate/ 1000; 
Nover2= N/ 2; % window overlap in 50% of frame size
K= N;
frame_window= hamming( N);
subframe_window= hamming( P); 
eta_v= .08; % used for gain values calc. 



% ====noise covariance matrix estimation
L120=floor( 120* Srate/ 1000);  
% assume the first 120ms is noise only
noise= noisy_speech( 1: L120);

noise_autoc= xcorr( noise, len- 1, 'biased');  
% from -(len- 1) to (len- 1)
% obtain the autocorrelation functions
Rn= toeplitz( noise_autoc( len: end));
% form a Toeplitz matrix to obtain the noise signal covariance matrix
bartlett_win= bartlett( 2* len- 1);
n_autoc_win= noise_autoc.* bartlett_win; 

for k= 0: N- 1
    Phi_w( k+ 1)= n_autoc_win( P: 2*P-1)' * 2* ...
        cos( 2* pi* k* (0: P- 1)'/ N)- n_autoc_win( P); 
end
Phi_w= Phi_w';
% make Phi_w column vector

n_start= 1;

Nframes= floor( length( noisy_speech)/ (N/ 2))- 1;  % number of frames  
x_overlap= zeros( Nover2, 1);

%===============================  Start Processing =====================


for n=1: Nframes 
    
    noisy= noisy_speech( n_start: n_start+ N- 1);     
    noisy_autoc= xcorr( noisy, len- 1, 'biased');
    Ry= toeplitz( noisy_autoc( len: 2* len- 1));    
    
    % Use simple VAD algorithm to update noise cov matrix, Rn
    %
    vad_ratio= Ry(1,1)/ Rn(1,1); 
    if (vad_ratio<= vad_thre) % noise dominant
        Rn= mu_vad* Rn+ (1- mu_vad)* Ry;  
        noise_autoc_sc= Rn( 1, :)';
        % single sided noise autocorrelation
        noise_autoc= [flipud( noise_autoc_sc( 2: end)); ...
            noise_autoc_sc];
        n_autoc_win= noise_autoc.* bartlett_win;
        % compute Phi_w
        for k= 0: N- 1
            Phi_w( k+ 1)= n_autoc_win( P: 2*P-1)' * 2* ...
                cos( 2* pi* k* (0: P- 1)'/ N)- n_autoc_win( P);
        end        
        % Phi_w is column vector 
    end
    % =================

    Rx= Ry- Rn;
    
    [U, D]= eig( Rx);
    dD= diag( D); % retrieving diagonal elements
    dD_Q= find( dD> 0); % index for those eigenvalues greater than 0
    Lambda= dD( dD_Q); 
    U1= U( :, dD_Q); 
    % eigenvector for those eigenvalues greater than 0   
    
    U1_fft= fft( U1, N); 
    V= abs( U1_fft).^ 2;     
    Phi_B= V* Lambda/ P;    
    
    %==calculating masking threshold
    Phi_mask= mask( Phi_B( 1: N/ 2+ 1), N, Srate, NBITS);
    Phi_mask= [Phi_mask; flipud( Phi_mask( 2: N/ 2))]; 
    
    Theta= V'* Phi_mask/ K; 
    Ksi= V'* Phi_w/ K; 
          
    gain_vals= exp( -eta_v* Ksi./ min( Lambda, Theta));    
    G= diag( gain_vals);
    H= U1* G* U1';
    
    % first step of synthesis for subframe
    sub_start= 1; 
    sub_overlap= zeros( P/2, 1);
    for m= 1: (2*N/P- 1)
        sub_noisy= noisy( sub_start: sub_start+ P- 1);
        enhanced_sub_tmp= (H* sub_noisy).* subframe_window;
        enhanced_sub( sub_start: sub_start+ P/2- 1)= ...
            enhanced_sub_tmp( 1: P/2)+ sub_overlap; 
        sub_overlap= enhanced_sub_tmp( P/2+1: P);
        sub_start= sub_start+ P/2;
    end
    enhanced_sub( sub_start: sub_start+ P/2- 1)= sub_overlap; 
        
    xi= enhanced_sub'.* frame_window;    
    xfinal( n_start: n_start+ Nover2- 1)= x_overlap+ xi( 1: Nover2);    
    x_overlap= xi( Nover2+ 1: N);               
        
    n_start= n_start+ Nover2; 
    
end

xfinal( n_start: n_start+ Nover2- 1)= x_overlap; 


wavwrite(xfinal, Srate, NBITS, outfile);


%=======================E N D ===================================

function M= mask( Sx, dft_length, Fs, nbits)
% Author: Patrick J. Wolfe
%         Signal Processing Group
%         Cambridge University Engineering Department
%         p.wolfe@ieee.org
% Johnston perceptual model initialisation
frame_overlap= dft_length/ 2;    
freq_val = (0:Fs/dft_length:Fs/2)';
half_lsb = (1/(2^nbits-1))^2/dft_length;

freq= freq_val;
thresh= half_lsb;
crit_band_ends = [0;100;200;300;400;510;630;770;920;1080;1270;...
        1480;1720;2000;2320;2700;3150;3700;4400;5300;6400;7700;...
        9500;12000;15500;Inf];

% Maximum Bark frequency
%
imax = max(find(crit_band_ends < freq(end)));

% Normalised (to 0 dB) threshold of hearing values (Fletcher, 1929) 
% as used  by Johnston.  First and last thresholds are corresponding 
% critical band endpoint values, elsewhere means of interpolated 
% critical band endpoint threshold values are used.
%
abs_thr = 10.^([38;31;22;18.5;15.5;13;11;9.5;8.75;7.25;4.75;2.75;...
        1.5;0.5;0;0;0;0;2;7;12;15.5;18;24;29]./10);
ABSOLUTE_THRESH = thresh.*abs_thr(1:imax);

% Calculation of tone-masking-noise offset ratio in dB
%
OFFSET_RATIO_DB = 9+ (1:imax)';

% Initialisation of matrices for bark/linear frequency conversion
% (loop increments i to the proper critical band)
%
num_bins = length(freq);
LIN_TO_BARK = zeros(imax,num_bins);
i = 1;
for j = 1:num_bins
    while ~((freq(j) >= crit_band_ends(i)) & ...
            (freq(j) < crit_band_ends(i+1))),
        i = i+1;
    end
    LIN_TO_BARK(i,j) = 1;
end

% Calculation of spreading function (Schroeder et al., 82)

spreading_fcn = zeros(imax);
summ = 0.474:imax;
spread = 10.^((15.81+7.5.*summ-17.5.*sqrt(1+summ.^2))./10);
for i = 1:imax
    for j = 1:imax
        spreading_fcn(i,j) = spread(abs(j-i)+1);
    end
end

% Calculation of excitation pattern function

EX_PAT = spreading_fcn* LIN_TO_BARK;

% Calculation of DC gain due to spreading function

DC_GAIN = spreading_fcn* ones(imax,1);


%Sx = X.* conj(X);

C = EX_PAT* Sx;

% Calculation of spectral flatness measure SFM_dB
%
[num_bins num_frames] = size(Sx);
k = 1/num_bins;
SFM_dB = 10.*log10((prod(Sx).^k)./(k.*sum(Sx)+eps)+ eps);

% Calculation of tonality coefficient and masked threshold offset
%
alpha = min(1,SFM_dB./-60);
O_dB = OFFSET_RATIO_DB(:,ones(1,num_frames)).*...
    alpha(ones(length(OFFSET_RATIO_DB),1),:) + 5.5;

% Threshold calculation and renormalisation, accounting for absolute 
% thresholds

T = C./10.^(O_dB./10);
T = T./DC_GAIN(:,ones(1,num_frames));
T = max( T, ABSOLUTE_THRESH(:, ones(1, num_frames)));

% Reconversion to linear frequency scale 

%M = 1.* sqrt((LIN_TO_BARK')*T);
M= LIN_TO_BARK'* T;
