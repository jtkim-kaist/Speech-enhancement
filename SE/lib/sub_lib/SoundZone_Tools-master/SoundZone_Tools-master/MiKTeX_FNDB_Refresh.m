function [ res, cmd_out ] = MiKTeX_FNDB_Refresh( )
% Function to refresh the File Name DataBase in MiKTeX
% 
% Syntax:	[ res, cmd_out ] = MiKTeX_FNDB_Refresh() This is used after
% exporting a file (i.e. figure) to a tex directory so that a direct comilation
% of a tex document will show the updated files.
% 
% Outputs: 
% 	res - Command exit status
% 	cmd_out - Output of the operating system command
% 
% Author: Jacob Donley
% University of Wollongong
% Email: jrd089@uowmail.edu.au
% Copyright: Jacob Donley 2017
% Date: 03 October 2015 
% Revision: 0.1
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
[res, cmd_out] = system('initexmf --update-fndb');
disp('Refreshed MiKTeX FNDB');

end

