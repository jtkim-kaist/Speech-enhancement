function d = taal2011(sigclean, sigproc, fs)
%TAAL2011  The Short-time objective intelligibility measure (STOI)
%   Usage: d = taal2011(sigclean, sigproc, fs);
%
%   d = stoi(sigclean, sigproc, fs) returns the output of the Short-Time
%   Objective Intelligibility (STOI) measure described in Taal
%   et. al. (2010) & (2011), where sigclean and sigproc denote the clean and
%   processed speech, respectively, with sample rate fs measured in
%   Hz. The output d is expected to have a monotonic relation with the
%   subjective speech-intelligibility, where a higher d denotes better
%   intelligible speech. See Taal et. al. (2010) & (2011) for more details.
%
%   The model consists of the following stages:
%
%   1) Removal of silent frames. Frames (of length 512) of the input
%      signals that have an energy of 40 dB less than the most energetic
%      frame are removed.
%
%   2) Expansion of the signals into a Fourier filterbank with a Hanning
%      window length of 25ms and 256 channels covering the the frequency
%      range from 0 to 5 kHz. The energy of the bands are then summed
%      into third-octaves
%
%   3) The output d is computed by a correlation process. See the
%      referenced papers for more details.
%
%   Examples:
%   ---------
%
%   The following example shows a simple comparison between the
%   intelligibility of a noisy speech signal and the same signal after
%   noise reduction using a simple soft thresholding (spectral
%   subtraction):
%  
%     % Get a clean and noisy test signal
%     [f,fs]=cocktailparty;
%     Ls=length(f);
%     f_noisy=f+0.05*pinknoise(Ls,1,'rms');
%
%     % Simple spectral subtraction to remove the noise
%     a=128; M=256; g=gabtight('hann',a,M);
%     c_noise   = dgtreal(f,g,a,M);
%     c_removed = thresh(c_noise,0.01);
%     f_removed = idgtreal(c_removed,g,a,M);
%     f_removed = f_removed(1:Ls);
%
%     % Compute the STOI of noisy vs. removed
%     d_noisy   = taal2011(f, f_noisy, fs)
%     d_removed = taal2011(f, f_removed, fs)
%
%   The original STOI model can be downloaded from
%   http://msp.ewi.tudelft.nl/content/short-time-objective-intelligibility-measure
%   This is a standalone version not depending on LTFAT and AMToolbox,
%   and licensed under a different license, but the models are
%   functionally equivalent.
%
%   References:
%     C. H. Taal, R. C. Hendriks, R. Heusdens, and J. Jensen. A Short-Time
%     Objective Intelligibility Measure for Time-Frequency Weighted Noisy
%     Speech. In Acoustics Speech and Signal Processing (ICASSP), pages
%     4214-4217. IEEE, 2010.
%     
%     C. H. Taal, R. C. Hendriks, R. Heusdens, and J. Jensen. An Algorithm
%     for Intelligibility Prediction of Time-Frequency Weighted Noisy Speech.
%     IEEE Transactions on Audio, Speech and Language Processing,
%     19(7):2125-2136, 2011.
%     
%
%   Url: http://amtoolbox.sourceforge.net/amt-0.9.5/doc/speech/taal2011.php

% Copyright (C) 2009-2014 Peter L. S��ndergaard.
% This file is part of AMToolbox version 1.0.0
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
  
  
%% -------- Model parameters ---------------------------------

fs_model = 10000; % Sample rate of proposed intelligibility measure
N_frame	= 256;    % Window support
K = 512;          % FFT size
J = 15;           % Number of 1/3 octave bands
mn = 150;         % Center frequency of first 1/3 octave band in Hz.
N = 30;           % Number of frames for intermediate intelligibility
                  % measure (Length analysis window)
Beta = -15;       % Lower SDR-bound
dyn_range = 40;   % Speech dynamic range

%% -------- Checking and initialization  ---------------------

% constant for clipping procedure
c           = 10^(-Beta/20);                            

if length(sigclean)~=length(sigproc)
    error('sigclean and sigproc should have the same length');
end

% Number of signals
W=size(sigclean,2);

% Get 1/3 octave band matrix
H = thirdoct(fs_model, K, J, mn);

% resample signals if other samplerate is used than fs_model The original
% model used the "resample" function. However, this function not part of the
% Matlab core functions and is not (at the time of writing) included in
% Octave.
if fs ~= fs_model
%   newlength= round(length(sigclean)/fs*fs_model);
%   sigclean = fftresample(sigclean, newlength);
%   sigproc  = fftresample(sigproc,  newlength);
  sigclean	= resample(sigclean, fs_model, fs);
  sigproc 	= resample(sigproc, fs_model, fs);
  
end

% Loop over multi-signals
d=zeros(1,W);
for w=1:W
  x=sigclean(:,w);
  y=sigproc(:,w);

  %% -------- Compute TF representation
  
  % Remove silent frames. Below, the clean signal is now called "x" and the
  % processed signal is called "y"
  [x,y] = removeSilentFrames(x,y,dyn_range, N_frame, N_frame/2);
  
  % Compute sampled short-time Fourier transforms using LTFAT
  x_hat=dgtreal(x,{'hann',N_frame},N_frame/2,K);
  y_hat=dgtreal(y,{'hann',N_frame},N_frame/2,K);  
  N_timesteps=size(x_hat, 2);
  
  % Collect the frequency bands into the 1/3 octave band TF-representation
  % This is done through multiplication with a sparse matrix consisting of
  % 0 and 1's
  X = sqrt(H*abs(x_hat).^2);
  Y = sqrt(H*abs(y_hat).^2);
  
  %% -------- Compute the intelligibility measure
  
  % loop al segments of length N and obtain intermediate intelligibility measure for all TF-regions
  
  % init memory for intermediate intelligibility measure
  d_interm  	= zeros(J, length(N:size(X, 2)));
    
  for m = N:size(X, 2)
    % regions with length N of clean and processed TF-units for all j
    X_seg = X(:, (m-N+1):m);
    Y_seg = Y(:, (m-N+1):m);
    
    % obtain scale factor for normalizing processed TF-region for all j
    alpha   = sqrt(sum(X_seg.^2, 2)./sum(Y_seg.^2, 2));
    
    % obtain \alpha*Y_j(n) from Eq.(2) [1], pointwise mul with alpha
    aY_seg 	= bsxfun(@times,Y_seg,alpha);
    for jj = 1:J
      % apply clipping from Eq.(3)   	
      Y_prime             = min(aY_seg(jj, :), X_seg(jj, :)+X_seg(jj, :)*c);
      
      % obtain correlation coeffecient from Eq.(4) [1]
      d_interm(jj, m-N+1)  = taa_corr(X_seg(jj, :).', Y_prime(:));
    end
  end
  
  % combine all intermediate intelligibility measures as in Eq.(4) [1]
  
  d(w) = mean(d_interm(:));                              
end;

%% ---------  Subfunctions -------------------------------


function  [A cf] = thirdoct(fs, N_fft, numBands, mn)
%   [A CF] = THIRDOCT(FS, N_FFT, NUMBANDS, MN) returns 1/3 octave band matrix
%   inputs:
%       FS:         samplerate 
%       N_FFT:      FFT size
%       NUMBANDS:   number of bands
%       MN:         center frequency of first 1/3 octave band
%   outputs:
%       A:          octave band matrix
%       CF:         center frequencies

f  = linspace(0, fs, N_fft+1);
f  = f(1:(N_fft/2+1));
k  = 0:(numBands-1); 
cf = 2.^(k/3)*mn;
fl = sqrt((2.^(k/3)*mn).*2.^((k-1)/3)*mn);
fr = sqrt((2.^(k/3)*mn).*2.^((k+1)/3)*mn);
%A  = spzeros(numBands, length(f));
A  = sparse(numBands, length(f));

for ii = 1:(length(cf))
  [a b] = min((f-fl(ii)).^2);
  fl(ii) = f(b);
  fl_ii = b;
  
  [a b] = min((f-fr(ii)).^2);
  fr(ii) = f(b);
  fr_ii = b;
  A(ii,fl_ii:(fr_ii-1))	= 1;
end

rnk         = sum(A, 2);
numBands = find((rnk(2:end)>=rnk(1:(end-1))) & (rnk(2:end)~=0)~=0, 1, 'last' )+1;
A           = A(1:numBands, :);
cf          = cf(1:numBands);

%%
%%
function [x_sil y_sil] = removeSilentFrames(x, y, range, N, K)
%   [X_SIL Y_SIL] = REMOVESILENTFRAMES(X, Y, RANGE, N, K) X and Y
%   are segmented with frame-length N and overlap K, where the maximum energy
%   of all frames of X is determined, say X_MAX. X_SIL and Y_SIL are the
%   reconstructed signals, excluding the frames, where the energy of a frame
%   of X is smaller than X_MAX-RANGE

x       = x(:);
y       = y(:);

frames  = 1:K:(length(x)-N);
w       = hanning(N);
msk     = zeros(size(frames));

for j = 1:length(frames)
    jj      = frames(j):(frames(j)+N-1);
    msk(j) 	= 20*log10(norm(x(jj).*w)./sqrt(N));
end

msk     = (msk-max(msk)+range)>0;
count   = 1;

x_sil   = zeros(size(x));
y_sil   = zeros(size(y));

for j = 1:length(frames)
    if msk(j)
        jj_i            = frames(j):(frames(j)+N-1);
        jj_o            = frames(count):(frames(count)+N-1);
        x_sil(jj_o)     = x_sil(jj_o) + x(jj_i).*w;
        y_sil(jj_o)  	= y_sil(jj_o) + y(jj_i).*w;
        count           = count+1;
    end
end

x_sil = x_sil(1:jj_o(end));
y_sil = y_sil(1:jj_o(end));


%%
function rho = taa_corr(x, y)
%   RHO = TAA_CORR(X, Y) Returns correlation coeffecient between column
%   vectors x and y. Gives same results as 'corr' from statistics toolbox.
xn    	= x-mean(x);
yn   	= y-mean(y);
rho   	= dot(xn/norm(xn),yn/norm(yn));