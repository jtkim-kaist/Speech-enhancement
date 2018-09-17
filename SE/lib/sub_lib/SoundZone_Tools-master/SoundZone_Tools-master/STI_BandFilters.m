function OctaveFilters = STI_BandFilters( N, fs )
% Calculation of the Speech Transmission Index (STI) Band Filters
% 
% Syntax:	OctaveFilters = STI_BandFilters( N, fs )
%       This function calculates the Speech Transmission Index octave band 
%       filters from 125Hz to 8kHz so they can be passed to the STI 
%       function for faster computation.
% 
% Inputs: 
% 	N - Filter order (recommended: 6)
% 	fs - Sampling frequency of the STI inputs
% 
% Outputs: 
% 	OctaveFilters - An array of octave band filters
% 
% Example: 
%   y=sinc(-7999:8000);
% 	fs=16000;
%   N = 6;
%   H = STI_BandFilters( N, fs )
%   tic;
% 	[STIval,ALcons,STIval_,ALcons_]=STI(y,fs,H)
%   toc
% 	[STIval,ALcons,STIval_,ALcons_]=STI(y,fs)
%   toc
% 
% See also: STI.m

% Author: Jacob Donley
% University of Wollongong
% Email: jrd089@uowmail.edu.au
% Version: 1.0 (12 April 2017)
% Version: 0.1 (30 September 2015)
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

BandsPerOctave = 1;
F0 = 1000;       % Center Frequency (Hz)
f = fdesign.octave(BandsPerOctave,'Class 1','N,F0',N,F0,fs);
F0 = validfrequencies(f);
F0(F0<125 | F0>min([8000,fs/2]))=[]; % Only keep bands in the range required for the STI calculation
Nfc = length(F0);

for i=1:Nfc
    f.F0 = F0(i);
    OctaveFilters(i) = design(f,'butter');    
end

end
