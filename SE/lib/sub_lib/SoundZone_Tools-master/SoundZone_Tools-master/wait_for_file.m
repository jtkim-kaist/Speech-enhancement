function wait_for_file( file_string )
% A forceful method to wait for a file to finish being written to
% 
% Syntax:	wait_for_file( file_string )
% 
% Inputs: 
% 	file_string - Absolute or relative path for the file to wait for
% 
%
% Author: Jacob Donley
% University of Wollongong
% Email: jrd089@uowmail.edu.au
% Copyright: Jacob Donley 2017
% Date: 03 October 2015 
% Revision: 0.1
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fid = fopen(file_string,'r');
while fid==-1
    fid = fopen(file_string,'r');
end
fseek(fid,0,'eof');
char_pos(1) = -1;
char_pos(2) = ftell(fid);
pause(0.5);
while char_pos(1) ~= char_pos(2)
    char_pos(1) = char_pos(2);
    fseek(fid,0,'eof');
    char_pos(2) = ftell(fid);
    pause(0.5);
end
fclose(fid);
end

