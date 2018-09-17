function [x_n] = addNoise( x, level, type )
% Adds a given level and type of noise to a signal
% 
% Syntax:	[x_n] = ADDNOISE( x, level, type ) Pass the signal, x, level
% and type of noise to the function and x_n will be the resulting signal
% with the noise added.
% 
% Inputs: 
% 	x - The input signal to add noise to
% 	level - The noise in dB relative to the input signal
% 	type - The type of noise to add
% 
% Outputs: 
% 	x_n - The noisey output signal
% 
% 
% Author: Jacob Donley
% University of Wollongong
% Email: jrd089@uowmail.edu.au
% Copyright: Jacob Donley 2017
% Date: 21 August 2015 
% Revision: 0.1
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x_n = Tools.generateNoise( x, level, type, true);

end
