function [stoi_PercCorr] = stoi_d2percCorr(stoi_d)
% Converts the stoi measure, d, to percent words correct unit
% 
% Syntax:	[stoi_PercCorr] = stoi_d2percCorr(stoi_d) 
% 
% Inputs: 
% 	stoi_d - The STOI measure in the raw units
% 
% Outputs: 
% 	stoi_PercCorr - The corressponding STOI value in Percent of Words
% 	Correct as defined for the IEEE English Library
%
% 
%    References:
%      C. H. Taal, R. C. Hendriks, R. Heusdens, and J. Jensen. A Short-Time
%      Objective Intelligibility Measure for Time-Frequency Weighted Noisy
%      Speech. In Acoustics Speech and Signal Processing (ICASSP), pages
%      4214-4217. IEEE, 2010.
%      
%      C. H. Taal, R. C. Hendriks, R. Heusdens, and J. Jensen. An Algorithm
%      for Intelligibility Prediction of Time-Frequency Weighted Noisy Speech.
%      IEEE Transactions on Audio, Speech and Language Processing,
%      19(7):2125-2136, 2011.

% Author: Jacob Donley
% University of Wollongong
% Email: jrd089@uowmail.edu.au
% Date: 05 August 2015 
% Revision: 0.1
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

f_IEEE_a = -17.4906;
f_IEEE_b = 9.6921;

stoi_PercCorr = 100 / (1 + exp(f_IEEE_a * stoi_d + f_IEEE_b ) );


end
