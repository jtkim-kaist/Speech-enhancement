function [ STI_val, ALcons, STI_val_approx, ALcons_approx ] = STI( ImpulseResponse, fs, OctaveFilters )
% Calculation of the Speech Transmission Index (STI)
% 
% Syntax:	[STI_val, ALcons, STI_val_approx, ALcons_approx] = STI(ImpulseResponse, fs, OctaveFilters)
%       This function calculates the Speech Transmission Index from a given
%       impulse response. Octave band filters (125Hz-8kHz) can be passed to
%       the function as an array for faster computation.
% 
% Inputs: 
% 	ImpulseResponse - Impulse response of the channel to test
% 	fs - Sampling frequency of the impulse response
% 	OctaveFilters - Array of MATLAB octave band filters
% 
% Outputs: 
% 	STI_val - The Speech Transmission Index result
% 	ALcons - Articulation loss of consonants result
% 	STI_val_approx - STI approximation from computable bands if fs too low
% 	ALcons_approx - ALcons approximation from computable bands if fs too low
% 
% Example: 
%   y=sinc(-7999:8000);
% 	fs=44100;
% 	[STIval,ALcons,STIval_,ALcons_]=STI(y,fs)
% 	fs=16000;
% 	[STIval,ALcons,STIval_,ALcons_]=STI(y,fs)
% 	fs=8000;
% 	[STIval,ALcons,STIval_,ALcons_]=STI(y,fs)
% 
% See also: STI_BandFilters.m, extractIR.m, synthSweep.m

% Author: Jacob Donley
% University of Wollongong
% Email: jrd089@uowmail.edu.au
% Version: 1.0 (12 April 2017)
% Version: 0.1 (30 September 2015)
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 1. Take the full bandwidth impulse response, run it through octave band filters
BandsPerOctave = 1;
N = 6;           % Filter Order
F0 = 1000;       % Center Frequency (Hz)
f = fdesign.octave(BandsPerOctave,'Class 1','N,F0',N,F0,fs);
F0 = validfrequencies(f);
F0(F0<125 | F0>min([8000,fs/2]))=[]; % Only keep bands in the range required for the STI calculation
Nfc = length(F0);

% Create MTF function handle
getMTF = @(ir)      ...
    abs(            ... Take the square root of the sum of the real part squared and the imaginary part squared
    fft(ir .^ 2)    ... Compute the Fourier transform of the squared impulse response
    / sum(ir .^ 2) );%  Integrate the squared impulse response to get the total energy and normalize the envelope spectrum

for i=1:Nfc
    if nargin < 3
        f.F0 = F0(i);
        H = design(f,'butter');
        ir_filtered = filter(H, ImpulseResponse);
    else
        ir_filtered = filter(OctaveFilters(i), ImpulseResponse);
    end
    MTF_octband(:,i) = getMTF(ir_filtered);
end

% 2. Take the amplitude at 14 modulation frequencies
%modulation_freqs = [ 0.63, 0.80, 1.00, 1.25, 1.60, 2.00, 2.50, 3.15, 4.00, 5.00, 6.30, 8.00, 10.00, 12.50];
modulation_freqs = 2 .^ ([-2:11]./3);
freqs = linspace(0, fs/2, size(MTF_octband,1)/2+1); freqs(end)=[];%No nyquist frequency
for i=1:Nfc
        m(i,:) = interp1(freqs,MTF_octband(1:end/2,i),modulation_freqs);
end
good_freqs = ~any(isnan(m),2);
m(~good_freqs,:)=[];

% 3. Convert each of the 98 m values into an “apparent signal-to-noise ratio” (S/N) in dB
SNR_apparent = pow2db( m ./ (1-m) );

% 4. Limit the Range
SNR_apparent( SNR_apparent > 15 ) = 15;
SNR_apparent( SNR_apparent < -15 ) = -15;

% 5. Compute the mean (S/N) for each octave band
SNR_avg = mean(SNR_apparent,2);

% 6. Weight the octave mean (S/N) values
W = [0.13, 0.14, 0.11, 0.12, 0.19, 0.17, 0.14]; % Values from the STI standard
SNR_Wavg = sum(SNR_avg' .* W(good_freqs));
SNR_Wavg_approx = sum(SNR_avg' .* W(good_freqs)/sum(W(good_freqs)));

% 7. Convert the overall mean (S/N) to an STI value
STI_val = (SNR_Wavg + 15) / 30;
STI_val_approx = (SNR_Wavg_approx + 15) / 30;

% (Optional) Convert STI to ALcons
sti2alcons = @(x) 170.5405*exp(-5.419*x);
ALcons = sti2alcons(STI_val);
ALcons_approx = sti2alcons(STI_val_approx);

end
