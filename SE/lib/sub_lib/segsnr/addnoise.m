function [ noisy, noise ] = addnoise( signal, noise, snr )
% ADDNOISE Add noise to signal at a prescribed SNR level.
%
%   [NOISY,NOISE]=ADDNOISE(SIGNAL,NOISE,SNR) adds NOISE to SIGNAL
%   at a prescribed SNR level. Returns the mixture signal as well 
%   as scaled noise such that NOISY=SIGNAL+NOISE.
%   
%   Inputs
%           SIGNAL is a target signal as vector.
%
%           NOISE is a masker signal as vector, such that 
%                 length(NOISE)>=length(SIGNAL). Note that 
%                 in the case that length(NOISE)>length(SIGNAL),
%                 a vector of length length(SIGNAL) is selected 
%                 from NOISE starting at a random sample number.
%           
%           SNR is the desired signal-to-noise ratio level (dB).
%
%   Outputs 
%           NOISY is a mixture signal of SIGNAL and NOISE at given SNR.
%
%           NOISE is a scaled masker signal, such that the mixture
%                 NOISY=SIGNAL+NOISE has the desired SNR.
%
%   Example
%           % inline function for SNR calculation
%           SNR = @(signal,noisy)( 20*log10(norm(signal)/norm(signal-noisy)) );
%   
%           fs   = 16000;                       % sampling frequency (Hz)
%           freq = 1000;                        % sinusoid frequency (Hz)
%           time = [ 0:1/fs:2 ];                % time vector (s)
%           signal = sin( 2*pi*freq*time );     % signal vector (s)
%           noise = randn( size(signal) );      % noise vector (s)
%           snr = -5;                           % desired SNR level (dB)
%
%           % generate mixture signal: noisy = signal + noise
%           [ noisy, noise ] = addnoise( signal, noise, snr ); 
%
%           % check the resulting signal-to-noise ratio
%           fprintf( 'SNR: %0.2f dB\n', SNR(signal,noisy) );
%
%   See also TEST_ADDNOISE_SINUSOID, TEST_ADDNOISE_SPEECH.

%   Author: Kamil Wojcicki, UTD, July 2011


    % inline function for SNR calculation
    SNR = @(signal,noisy)( 20*log10(norm(signal)/norm(signal-noisy)) );

    % needed for older realases of MATLAB
    randi = @(n)( round(1+(n-1)*rand) );

    % ensure masker is at least as long as the target
    S = length( signal );
    N = length( noise );
    if( S>N ), error( 'Error: length(signal)>length(noise)' ); end;

    % generate a random start location in the masker signal
    R = randi(1+N-S);

    % extract random section of the masker signal
    noise = noise(R:R+S-1);

    % scale the masker w.r.t. to target at a desired SNR level
    noise = noise / norm(noise) * norm(signal) / 10.0^(0.05*snr);

    % generate the mixture signal
    noisy = signal + noise;

    % sanity check
    assert( abs(SNR(signal,noisy)-snr) < 1E10*eps(snr) ); 


%%% EOF
