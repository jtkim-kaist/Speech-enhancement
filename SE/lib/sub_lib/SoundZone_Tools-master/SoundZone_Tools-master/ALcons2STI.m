function [ STIval ] = ALcons2STI( ALcons )
% Converts Articulation Loss of Consonants (ALcons) to the Speech Transmission Index (STI)
% 
% Syntax:	[STIval] = ALcons2STI( ALcons )
%       Pass the Articulation Loss of Consonants (ALcons) value to the
%       function to retrieve the STI value.
% 
% Inputs: 
% 	ALcons - Articulation Loss of Consonants value
% 
% Outputs: 
% 	STIval - Corresponding Speech Transmission Index value
% 
% 
% Author: Jacob Donley
% University of Wollongong
% Email: jrd089@uowmail.edu.au
% Version: 1.0 (12 April 2017)
% Version: 0.1 (30 September 2015)
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

STIval = -0.1845 * log(ALcons) + 0.9482;

end

