function [ invirfft ] = invSweepFFT( y, f1, f2, fs, Nfft )
% Obtain the FFT of an inverted exponentional sine sweep
%   Detailed explanation goes here
if nargin < 5
    Nfft = length(y);
end

%%% get fft of sweep to verify that its okay and to use for inverse
yfft = fft(y(:)', Nfft);

% start with the true inverse of the sweep fft
invirfft = 1./yfft;


%%% Create band pass magnitude to start and stop at desired frequencies
[B1, A1] = butter(2,f1/(fs/2),'high' );  %%% HP at f1
[B2, A2] = butter(2,f2/(fs/2));          %%% LP at f2

%%% so we need to re-apply the band pass here to get rid of that
H1 = freqz(B1,A1,length(yfft),fs,'whole');
H2 = freqz(B2,A2,length(yfft),fs,'whole');

%%% apply band pass filter to inverse magnitude
invirfft = invirfft .* abs(H1)' .* abs(H2)';

end

