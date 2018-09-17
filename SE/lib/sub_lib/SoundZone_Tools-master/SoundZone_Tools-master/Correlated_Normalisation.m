function [ X_Approx_Adjust, Scaling_Factor ] = correlated_normalisation( X, X_Approx )
% Matches the amplitude of X using cross-correlation
% 
% Syntax:	[ X_Approx_Adjust, Scaling_Factor ] = CORRELATED_NORMALISATION( X, X_Approx )
% 
% Inputs: 
% 	X - Original signal
% 	X_Approx - Approximation of the original signal
% 
% Outputs: 
% 	X_Approx_Adjust - The approximate signal with a magnitude closely
% 	matching the original signal
% 	Scaling_Factor - The adjustment factor required
 
% Author: Jacob Donley
% University of Wollongong
% Email: jrd089@uowmail.edu.au
% Copyright: Jacob Donley 2017
% Date: 19 June 2015 
% Revision: 0.1
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


corr = xcorr( X, X_Approx);
peakVal = max(abs(corr));
Energy = sum( X .^ 2 );

if isnan(peakVal)
    peakVal = mean([max(abs(X)), max(abs(X_Approx))]);
end

Scaling_Factor = peakVal / Energy;

X_Approx_Adjust = X_Approx * 1/Scaling_Factor;

end

