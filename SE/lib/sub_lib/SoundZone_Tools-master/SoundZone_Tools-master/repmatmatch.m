function A = repmatmatch(a,B)
%Replicate and tile an array to match the size of a given N-D array
% 
% Syntax:	A = repmatmatch(a,B)
% 
% Inputs: 
% 	a - Input array to tile
% 	B - N-D array to match the size of
% 
% Outputs: 
% 	A - The replicated and tiled copy of input 'a'
% 
% Example: 
% 	a = [1 2 3];
% 	B = rand(2,3);
% 	A = repmatmatch(a,B);
% 
% See also: repmat

% Author: Jacob Donley
% University of Wollongong
% Email: jrd089@uowmail.edu.au
% Copyright: Jacob Donley 2017
% Date: 28 August 2017 
% Version: 0.1 (28 August 2017)
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

aSz = size(a);
BSz = size(B);

if any(rem(BSz ./ aSz,1))
    error(['The size of each dimension of ''a'' should match ''B'' ' ...
       'or be evenly divisible by the corresponding dimension of ''B''.'])
end

A = repmat(a, BSz ./ aSz);

end
