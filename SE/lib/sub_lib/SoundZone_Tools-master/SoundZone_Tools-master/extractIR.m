function [irLin, irNonLin] = extractIR(sweep_response, invsweepfft)
% Extract impulse response from swept-sine response.
%   [irLin, irNonLin] = extractIR(sweep_response, invsweepfft) 
%   Extracts the impulse response from the swept-sine response.  Use
%   synthSweep.m first to create the stimulus; then pass it through the
%   device under test; finally, take the response and process it with the
%   inverse swept-sine to produce the linear impulse response and
%   non-linear simplified Volterra diagonals.  The location of each
%   non-linear order can be calculated with the sweepRate - this will be
%   implemented as a future revision.
%   
%   Developed at Oygo Sound LLC
%
%   Equations from Muller and Massarani, "Transfer Function Measurement
%   with Sweeps."
%
%   Modified by Jacob Donley (jrd089@uowmail.edu.au) (January 2017)

if diff(size(sweep_response))>0, sweep_response = sweep_response.'; end
if diff(size(invsweepfft))>0, invsweepfft = invsweepfft.'; end

N = length(invsweepfft);
sweepfft = fft(sweep_response,N);

%%% convolve sweep with inverse sweep (freq domain multiply)
invsweepfft = repmat(invsweepfft,1,size(sweepfft,2));
ir = real(ifft(invsweepfft.*sweepfft));

ir = circshift(ir, length(ir)/2, 1); 

irLin = ir(end/2+1:end,:);
irNonLin = ir(1:end/2,:);
