function [ res ] = pesq3( reference_sig, degraded_sig, Fs, fileNum )
% A wrapper for the objective Perceptual Evaluation of Speech Quality measure
% 
% Syntax:	[ res ] = pesq3( reference_sig, degraded_sig, Fs, fileNum )
% 
% Inputs: 
% 	reference_sig - Reference (clean, talker, sender) speech signal
% 	degraded_sig - Degraded (noisy, listener, receiver) speech signal
% 	Fs - Sampling Frequency
%   fileNum - An ID number to append to the temporary audio files. Useful
%   when several instances are to be run together (in parallel).
% 
% Outputs: 
% 	res - Raw PESQ result for narrowband and MOS-LQO result for wideband
% 
% See also: pesq2mos.m

% Author: Jacob Donley
% University of Wollongong
% Email: jrd089@uowmail.edu.au
% Copyright: Jacob Donley 2017
% Date: 03 October 2015 
% Revision: 0.1
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin < 4
    fileNum = 0;
end
temp_path = [pwd filesep '+Miscellaneous/+Temporary/'];
ref_path = [ 'tmp_ref' num2str(fileNum) '.wav'];
deg_path = [ 'tmp_deg' num2str(fileNum) '.wav'];

if ~exist(temp_path,'dir'); mkdir(temp_path); end

max_val = max(abs([reference_sig(:); degraded_sig(:)]));

audiowrite([temp_path ref_path], reference_sig / max_val, Fs);
audiowrite([temp_path deg_path], degraded_sig / max_val, Fs);

res = pesq2_mtlb(ref_path, ...
                 deg_path, ...
                 Fs, 'wb', [pwd filesep '+Tools/pesq_NoResFile.exe'], ...
                 temp_path);

end

