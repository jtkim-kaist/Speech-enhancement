function [ y_shaped ] = shapeSpectrum( y, spectrum, freqs, fs )
% This function will shape an input signal to the given spectrum (simple, unregulated spectral shaping)
% 
% Syntax:	[ y_shaped ] = shapeSpectrum( y, fs, spectrum )
% 
% Inputs: 
% 	y - Input signal in the time-domain which requires shaping.
% 	spectrum - A vector containing the magnitude spectrum to shape the
% 	signal with. ( length = fs/2 - 1 )
%   freqs - A vector containing the corresponding frequencies for the
%   spectrum vector.
% 
% Outputs: 
% 	y_shaped - The input signal shaped to the given spectrum
% 
% 

% Author: Jacob Donley
% University of Wollongong
% Email: jrd089@uowmail.edu.au
% Copyright: Jacob Donley 2017
% Date: 26 February 2016
% Revision: 0.1
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N = length( y );

if rem(N,2)
    M = N+1;
else
    M = N;
end

% Frequency Domain Tranform
Y = fft(y); % Fast Fourier Transform

% Frequencies
NumPts = M/2 + 1;
freqs_new = linspace(0, fs/2, NumPts);

% Create correct size spectrum vector
spect = interp1( freqs, spectrum, freqs_new )';

% Apply magnitude weighting
Y(1:NumPts) = Y(1:NumPts) .* spect;

% Apply conjugation for negative frequency side of spectrum
Y(NumPts+1:M) = conj(Y(M/2:-1:2));

% Time Domain Transform
y_shaped = ifft(Y); % Inverse Fast Fourier Transform

% prepare output vector y
y_shaped = real(y_shaped(1:N));

% ensure unity standard deviation and zero mean value
y_shaped = y_shaped - mean(y_shaped);
y_shaped = y_shaped / rms(y_shaped);


% BETTER METHOD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end
