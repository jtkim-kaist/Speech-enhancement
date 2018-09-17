function [ interpolated_values,  interpolated_indices] = interpVal( values, index_values, desired_index_values )
% This function will interpolate from desired arbitrarily spaced index values
%
% Syntax:	[ interpolated_values,  interpolated_indices] = interpVal( values, index_values, desired_index_values )
%
% Inputs:
% 	values - A 1D array of values to interpolate between
% 	index_values - The axis values of the array
%   desired_index_values - The desired axis values to interpolate to
%   (Can be spaced abitrarily)
%
% Outputs:
%   interpolated_values - The new interpolated values
% 	interpolated_indices - The new interpolated indices
%
% Example:
%
% See also: interp1

% Author: Jacob Donley
% University of Wollongong
% Email: jrd089@uowmail.edu.au
% Copyright: Jacob Donley 2015-2017
% Date: 15 August 2017
% Version: 0.2 (15 August 2017)
% Version: 0.1 (03 October 2015)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% center desireable value
temp = repmat(index_values, [length(desired_index_values) 1]) ...
    - repmat(desired_index_values', [1 length(index_values)]); 
temp(temp > 0) = NaN;

N = numel(index_values);
[~, i_low] = min(abs(temp), [], 2);
i_low  = (i_low - (i_low>N).*(i_low-N))';
i_high =  i_low + (N~=1);
i_high(i_high>N) = N;

num = (desired_index_values - index_values(i_low));
den = (index_values(i_high) - index_values(i_low));

div = num ./ den;
div(num==0&den==0)=0; % redefine zero devided by zero as 0/0=0 

interpolated_indices = div + i_low;

interpolated_values = interp1(values, interpolated_indices);

end

