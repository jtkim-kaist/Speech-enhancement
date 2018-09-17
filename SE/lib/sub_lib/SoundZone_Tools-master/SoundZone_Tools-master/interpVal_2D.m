function [ interpolated_values ] = interpVal_2D( values, index_values1, index_values2, desired_index_values1, desired_index_values2, interpolation_type )
% This function will interpolate from desired abitrarily spaced index values in a 2D array
% 
% Syntax:	[ interpolated_values ] = interpVal_2D( ...
%               values, ...
%               index_values1, index_values2, ...
%               desired_index_values1, desired_index_values2, ...
%               interpolation_type )
% 
% Inputs: 
% 	               values - A 2D matrix of Z-axis values to interpolate
%                           between
%       	index_values1 - The x-axis values of the matrix
% 	        index_values2 - The y-axis values of the matrix
%   desired_index_values1 - The desired x-axis values to interpolate to
%                           (Can be spaced abitrarily)
%   desired_index_values2 - The desired y-axis values to interpolate to
%                           (Can be spaced abitrarily)
%      interpolation_type - An interpolation type supported by interp2
%                           (e.g. 'linear', 'nearest', 'cubic' or 'spline')
% 
% Outputs: 
%     interpolated_values - The new interpolated values
% 
% Example: 
% 
% See also: interp2

% Author: Jacob Donley
% University of Wollongong
% Email: jrd089@uowmail.edu.au
% Copyright: Jacob Donley 2015-2017
% Date: 03 October 2015
% Revision: 0.1 (03 October 2015)
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin < 6
    interpolation_type = 'linear';
end

%% Just incase this function tries to call interpVal within a class folder
% we should create a function handle for interpVal regardless
inf = dbstack('-completenames');
funcName = 'interpVal';
funcPath = inf.file;
classDirs = getClassDirs(funcPath);
interpVal_ = str2func([classDirs funcName]);


%% Start interpolation procedure
interpolated_indices1 = zeros(length(desired_index_values1) + 1,1);
interpolated_indices2 = zeros(length(desired_index_values2) + 1,1);

[~, interpolated_indices1] = interpVal_(interpolated_indices1, index_values1, desired_index_values1);
[~, interpolated_indices2] = interpVal_(interpolated_indices2, index_values2, desired_index_values2);

if any(diff(interpolated_indices1)) && any(diff(interpolated_indices2))
    % if interpolation is possible over both dimensions
    interpolated_values = interp2(values, ...
        interpolated_indices1, interpolated_indices2, interpolation_type);
    
elseif ~any(diff(interpolated_indices1))
    % if interpolation is not possible along 1st dimension
    interpolated_values = interp1(values(:,interpolated_indices1(1)), ...
        interpolated_indices2, interpolation_type);
    
elseif ~any(diff(interpolated_indices2))
    % if interpolation is not possible along 2nd dimension
    interpolated_values = interp1(values(interpolated_indices2(1),:), ...
        interpolated_indices1, interpolation_type);
    
end
    

end

function classDirs = getClassDirs(FullPath)
    classDirs = '';
    classes = strfind(FullPath,'+');
    for c = 1:length(classes)
        clas = FullPath(classes(c):end);
        stp = strfind(clas,filesep);
       classDirs = [classDirs  clas(2:stp(1)-1) '.'];
    end
end