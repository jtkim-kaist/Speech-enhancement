function y = ArbitraryOctaveFilt(x, SPECT, FREQS, N, fs, octBandwidth)
% Filters a signal with any arbitrary spectrum smoothed with any fractional octave band average
% 
% Syntax:	Y = ARBITRARYOCTAVEFILT(X, SPECT, FREQS, N, FS, OCTBANDWIDTH)
% 
% Inputs: 
% 	x - Input signal to filter as a vector
% 	SPECT - The spectrum to shape the input signal to
% 	FREQS - The frequencies of each SPECT element
% 	N - The length of the filter to usee
% 	fs - Description
% 	octBandwidth - Description
% 
% Outputs: 
% 	y - Description
%
% Example: 
%   fs = 16000;
%   T = 10;
%   N = 1000;
%   f = linspace(0,fs/2,N);
%   s = 1./f;
%   x = wgn(T*fs,1,0);
%   y = ArbitraryOctaveFilt(x,s,f,N,fs,1/3);
%   pwelch([x y]);
% 
% See also: fir2, filter

% Author: Jacob Donley
% University of Wollongong
% Email: jrd089@uowmail.edu.au
% Copyright: Jacob Donley 2017
% Date: 06 June 2016 
% Revision: 0.1
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin < 6
    octBandwidth = 1/6;    
end

% Find nth-octave averages
[MAG,f]=Tools.octaveBandMean(SPECT,FREQS,octBandwidth);

% Force even length filter
if isempty(N), if mod(length(SPECT),2), N=length(SPECT)-1; else N=length(SPECT); end; end
% Design arbitrary magnitude (linear-phase) filter
b = fir2(N,f/(fs/2),MAG);
% Apply filter
y = filter(b,1,x);


end