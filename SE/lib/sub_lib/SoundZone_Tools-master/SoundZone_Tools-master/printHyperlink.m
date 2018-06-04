function printHyperlink( URL, OptionalText )
% Prints a hyperlink to the command window
% 
% Syntax:	PRINTHYPERLINK( URL, OPTIONALTEXT)
% 
% Inputs: 
% 	URL - Address for the hyperlink
% 	OptionalText - The text to be printed which is hyperlinked
% 
% See also: fprintf, cprintf

% Author: Jacob Donley
% University of Wollongong
% Email: jrd089@uowmail.edu.au
% Copyright: Jacob Donley 2017
% Date: 16 February 2017 
% Revision: 0.2 (30 March 2017)
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin < 2
    OptionalText = URL;
end
fprintf(['<a href="' URL '" rel="nofollow">' OptionalText '</a>\n']);
if exist('cprintf','file')
    cprintf('hyper', '');
end

end
