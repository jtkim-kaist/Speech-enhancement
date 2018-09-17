function [ interpolated_index_values2 ] = interpFromVal_2D( values, index_values1, index_values2, desired_index_values1, desired_values2 )
% This function will interpolate from desired z-axis values and return the interpolation indices for them in the y-axis
% 
% Syntax:	[ interpolated_values ] = interpFromVal_2D( values, index_values1, index_values2, desired_index_values1, desired_index_values2 )
% 
% Inputs: 
% 	values - A 2D matrix of Z-axis values to interpolate between
% 	index_values1 - The x-axis values of the matrix
% 	index_values2 - The y-axis values of the matrix
%   desired_index_values1 - The desired x-axis values to interpolate to
%   (Can be spaced abitrarily)
%   desired_values2 - The desired z-axis values to interpolate to
%   (Can be spaced abitrarily)
% 
% Outputs: 
% 	interpolated_index_values2 - The new interpolated index values for the
% 	y-axis
% 
% Example: 
% 
% See also: List related files here

% Author: Jacob Donley
% University of Wollongong
% Email: jrd089@uowmail.edu.au
% Copyright: Jacob Donley 2017
% Date: 03 October 2015 
% Revision: 0.1
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Just incase this function tries to call interVal within a class folder we
% should create a function handle for interpVal regardless
inf = dbstack('-completenames');
funcName = 'interpVal';
funcPath = inf.file;
classDirs = getClassDirs(funcPath);
interpVal_ = str2func([classDirs funcName]);


%% Start interpolation procedure

interpolated_indices = zeros(length(desired_index_values1),1);
interpolated_index_values2  = zeros(length(desired_index_values1),1);

[~, interpolated_indices] = interpVal_(interpolated_indices, index_values1, desired_index_values1);

% for div = 1:length(desired_index_values1)
%     temp_ = interp2(values, interpolated_indices(div), 1:length(index_values2)); %interpolate data
%     temp = temp_ - desired_index_values2(div); % center desireable value
%     temp(temp >= 0) = NaN;
%     if sum(isnan(temp)) ~= length(temp)
%         [~, i_low] = min(abs(temp));
%     else
%         i_low = length(temp);
%     end
%     if i_low ~= length(temp)
%         i_high = i_low+1;
%         interpolated_values(div) = max( interp1(index_values2,  (desired_index_values2(div) - temp_(i_low)) / (temp_(i_high) - temp_(i_low)) + i_low  ), index_values2(1));
%     else
%         interpolated_values(div) = index_values2(i_low);
%     end
% end
    
    temp_ = interp2(values, interpolated_indices, 1:length(index_values2)); %interpolate data
    temp = temp_ - repmat(desired_values2, [size(temp_, 1) 1]); % center desireable value
    temp(temp >= 0) = NaN;
    
    i_low = zeros(1, size(temp,2));    
    [~, i_low(sum(isnan(temp), 1) ~= size(temp, 1))] = min(abs( temp(:, sum(isnan(temp), 1) ~= size(temp, 1))), [], 1);
    i_low(sum(isnan(temp), 1) == size(temp, 1)) = size(temp, 1);
    i_high = i_low(i_low ~= size(temp, 1)) + 1;
    
    low = diag(temp_(i_low(i_low ~= size(temp, 1)), i_low ~= size(temp, 1)))';
    high = diag(temp_(i_high,i_low ~= size(temp, 1)))';
    
    interpolated_index_values2(i_low ~= size(temp, 1)) = max( interp2( repmat(index_values2, [size(temp_,2) 1]), ...
        (desired_values2(i_low ~= size(temp, 1)) - low) ./ ...
        ( high - low ) + i_low(i_low ~= size(temp, 1)), 1:size(low, 2))  , index_values2(1));
        
    interpolated_index_values2(i_low == size(temp, 1)) = index_values2(i_low(i_low == size(temp, 1)));
    
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
