function [sweep, invsweepfft, sweepRate] = synthSweep(T,FS,f1,f2,tail,AsdB)
% Synthesize a logarithmic sine sweep
%   [sweep invsweepfft sweepRate] = SYNTHSWEEP(T,FS,f1,f2,tail,AsdB) 
%   generates a logarithmic sine sweep that starts at frequency f1 (Hz),
%   stops at frequency f2 (Hz) and duration T (sec) at sample rate FS (Hz).
%    
%   usePlots indicates whether to show frequency characteristics of the
%   sweep, and the optional AsdB is an amplitude suppression value in
%   decibels to avoid clipping
%   
%   Developed at Oygo Sound LLC
%
%   Equations from Muller and Massarani, "Transfer Function Measurement
%   with Sweeps."
%
%   Modified by Jacob Donley (jrd089@uowmail.edu.au) Nov 2016


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%             DO SOME PREPARATORY STUFF
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%% number of samples / frequency bins
N = real(round(T*FS));

if nargin < 5
    tail = 0;
end
if nargin < 6
   AsdB = 0; 
end


%%% make sure start frequency fits in the first fft bin
f1 = ceil( max(f1, FS/(2*N)) );

%%% set group delay of sweep's starting freq to one full period length of
%%% the starting frequency, or N/200 if thats too small, or N/10 if its too
%%% big
Gd_start = ceil(min(N/10,max(FS/f1, N/200)));

%%% set fadeout length
postfade = ceil(min(N/10,max(FS/f2,N/200)));

%%% find the length of the actual sweep when its between f1 and f2
Nsweep = N - tail - Gd_start - postfade;

%%% length in seconds of the actual sweep 
tsweep = Nsweep/FS;

sweepRate = log2(f2/f1)/tsweep;

%%% make a frequency vector for calcs (This  has length N+1) )
f = ([0:N]*FS)/(2*N);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%             CALCULATE DESIRED MAGNITUDE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% create pink (-10dB per decade, or 1/(sqrt(f)) spectrum
mag = [sqrt(f1./f(1:end))];
mag(1) = mag(2);

%%% Create band pass magnitude to start and stop at desired frequencies
[B1, A1] = butter(2,f1/(FS/2),'high' );  %%% HP at f1
[B2, A2] = butter(2,f2/(FS/2));          %%% LP at f2

%%% convert filters to freq domain
H1 = freqz(B1,A1,N+1,FS);          
H2 = freqz(B2,A2,N+1,FS);

%%% multiply mags to get final desired mag spectrum 
mag = mag .* abs(H1)' .* abs(H2)';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%            CALCULATE DESIRED GROUP DELAY
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% calc group delay for arbitrary mag spectrum with constant time envelope
%%% from Muller eq's 11 and 12
C = tsweep ./ sum(mag.^2);
Gd = C * cumsum(mag.^2);
Gd = Gd + Gd_start/FS; % add predelay
Gd = Gd*FS/2;   % convert from secs to samps

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%            CALCULATE DESIRED PHASE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% if (nargin > 6)
%     mag = linspace(0.1, 1, length(mag));
% end


%%% integrate group delay to get phase
ph = grpdelay2phase(Gd);

%%% force the phase at FS/2 to be a multiple of 2pi using Muller eq 10
%%% (but ending with mod 2pi instead of zero ...)
ph = ph - (f/(FS/2))*mod(ph(end),2*pi);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%             SYNTHESIZE COMPLEX FREQUENCY RESPONSE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cplx = mag.*exp(sqrt(-1)*ph); %%% put mag and phase together in polar form
cplx = [cplx conj(fliplr(cplx(2:end-1)))]; %%% conjugate, flip, append for whole spectrum

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%             EXTRACT IMPULSE RESPONSE WITH IFFT AND WINDOW
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ir = real(ifft(cplx));
err = max(abs(imag(ifft(cplx))));  %%% if this is not really tiny then something is wrong

%%% create window for fade-in and apply
w = hann(2*Gd_start)';
I = 1:Gd_start;
ir(I) = ir(I).*w(I);

%%% create window for fade-out and apply
w = hann(2*postfade)';
I = Gd_start+Nsweep+1:Gd_start+Nsweep+postfade;
ir(I) = ir(I).*w(postfade+1:end);

%%% force the tail beyond the fadeout to zeros
I = Gd_start+Nsweep+postfade+1:length(ir);
ir(I) = zeros(1,length(I));

%%% cut the sweep down to its correct size
ir = ir(1:end/2);

%%% normalize
ir = ir / (max(abs(ir(:))));

%%% suppress output
ir = ir * db2mag( AsdB );

%%% get fft of sweep to verify that its okay and to use for inverse
irfft = fft(ir);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%             CREATE INVERSE SPECTRUM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% start with the true inverse of the sweep fft
%%% this includes the band-pass filtering, whos inverse could go to
%%% infinity!!!
invirfft = 1./irfft;

%%% so we need to re-apply the band pass here to get rid of that
H1 = freqz(B1,A1,length(irfft),FS,'whole');
H2 = freqz(B2,A2,length(irfft),FS,'whole');

%%% apply band pass filter to inverse magnitude
% invirfftmag  = abs(invirfft).*abs(H1)'.*abs(H2)';
% 
% %%% get inverse phase
% invirfftphase = angle(invirfft);
% 
% %%% re-synthesis inverse fft in polar form
% invirfft = invirfftmag.*exp(sqrt(-1)*invirf
invirfft = invirfft .* abs(H1)' .* abs(H2)';

%%% assign outputs
invsweepfft = invirfft;
sweep = ir;






