function [SNR] = snr_mean( target, masked )
% SNR Computes signal-to-noise ratio.
%
%   S=SNR(TARGET,MASKED) returns signal-to-noise ratio 
%   given target and masked speech signals.
%   
%   Inputs
%           TARGET is a target signal as vector.
%
%           MASKED is a target+masker signal as vector.
%
%   Outputs 
%           S is the signal-to-noise ratio (dB).
%
%   Example
%           % read target and masker signals from wav files
%           [ target, fs ] = wavread( 'sp10.wav' );
%           [ masker, fs ] = wavread( 'ssn.wav' );
%
%           % desired SNR level (dB)
%           dSNR = 5; 
%
%           % generate mixture signal: noisy = signal + noise
%           [ masked, masker ] = addnoise( target, masker, dSNR ); 
%
%           % compute SNR (dB)
%           SNR = snr( target, masked );
%
%           % display the result 
%           fprintf( 'SNR: %0.2f dB\n', SNR );
%
%   See also ADDNOISE, SEGSNR.

%   Author: Kamil Wojcicki, November 2011.

    % compute the masker (assumes additive noise model)
    masker = masked(:) - target(:); 

    % compute target and masker frame energies
    energy.target = target.' * target;
    energy.masker = masker.' * masker + eps;

    % compute frame signal-to-noise ratio (dB)
    SNR = 10*log10( energy.target ./ energy.masker + eps );
    SNR
%     SNR = 123123123;


% EOF
