function [E,Eps] = COSHdist(H,P)
% Finds the symmetric Itakura-Saito distance using the hyperbolic cosine function
% 
% Syntax:	[OUTPUTARGS] = COSHDIST(INPUTARGS) Explain usage here
% 
% Inputs: 
% 	input1 - Description
% 	input2 - Description
% 	input3 - Description
% 
% Outputs: 
% 	output1 - Description
% 	output2 - Description
% 
% Example: 
% 	Line 1 of example
% 	Line 2 of example
% 	Line 3 of example
% 
% See also: List related files here

% Author: Jacob Donley
% University of Wollongong
% Email: jrd089@uowmail.edu.au
% Copyright: Jacob Donley 2017
% Date: 24 October 2016 
% Revision: 0.1
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[~,X]=size(P);
Hx = repmat(H(:),1,X);

% K = numel(H);
% E = 1/K *...
%      sum(  ...
%      abs(Hx)./P ...
%      - log( abs(Hx)./P ) ...
%      + P./abs(Hx) ...
%      - log( P./abs(Hx) ) ...
%      - 2 ) / 2;
 
E = mean( cosh( log( Hx./P ) ) - 1 );
 
 Eps = mean(E);

end
