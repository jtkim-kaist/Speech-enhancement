function [ res, cmd_out ] = Dropbox( action )
% Function to start and kill dropbox from MATLAB
% 
% Syntax:   [ res, cmd_out ] = Dropbox( action )
% 
% Inputs: 
% 	action - Action to perform. Either 'start' or 'kill.
% 
% Outputs: 
% 	res - Command exit status
% 	cmd_out - Output of the operating system command
% 
% Author: Jacob Donley
% University of Wollongong
% Email: jrd089@uowmail.edu.au
% Copyright: Jacob Donley 2017
% Date: 3 October 2015 
% Revision: 0.1
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

switch lower(action)
    case 'start'
        [res, cmd_out] = dos('start "" "%APPDATA%\Dropbox\bin\Dropbox.exe" & exit');
        disp('... Started Dropbox');
        
    case 'kill'
        [res, cmd_out] = dos('taskkill /F /IM Dropbox.exe');
        if res==0, disp('Terminated Dropbox ...'); end
        
    otherwise
        error('Dropbox action not supported...');
end

end

