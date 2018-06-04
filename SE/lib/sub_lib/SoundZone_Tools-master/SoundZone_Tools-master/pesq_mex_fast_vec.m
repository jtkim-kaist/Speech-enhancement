function [ res ] = pesq_mex_fast_vec( reference_sig, degraded_sig, Fs, modeOfOperation )
% Accepts vectors for a mex compiled version of the objective Perceptual Evaluation of Speech Quality measure
% 
% Syntax:	[ res ] = pesq_mex_vec( reference_sig, degraded_sig, Fs )
% 
% Inputs: 
% 	  reference_sig - Reference (clean, talker, sender) speech signal
% 	   degraded_sig - Degraded (noisy, listener, receiver) speech signal
% 	             Fs - Sampling Frequency
%   modeOfOperation - This optional string argument is used to specify
%                     whether the PESQ mex function runs in narrowband,
%                     wideband or both. The possible values are 
%                     'narrowband', 'wideband' (default) or 'both'.
% 
% Outputs: 
% 	res - MOS-LQO result for given modeOfOperation (wideband by default). 
%         If 'both' is specified as the modeOfOperation then res is a
%         column vector of the format [narrowband_result; wideband_result].
% 
% See also: pesq2mos.m

% Author: Jacob Donley
% University of Wollongong
% Email: jrd089@uowmail.edu.au
% Copyright: Jacob Donley 2017
% Date: 2 August 2017
% Revision: 0.3 (2 August 2017)
% Revision: 0.2 (16 June 2016)
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Just incase this function tries to call within a class folder we should 
% create a function handle for this function to use instead
infun = dbstack('-completenames');
funcName = 'pesq_mex_fast';
funcPath = infun.file;
classDirs = getClassDirs(funcPath);
pesq_mex_ = str2func([classDirs funcName]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin < 4
    modeOfOperation = 'wideband';
end

switch lower(modeOfOperation)
    case 'narrowband'
        mOp = {''};
    case 'wideband'
        mOp = {'+wb'};
    case 'both'
        mOp = {'';'+wb'};
    otherwise
        error(['''' modeOfOperation ''' is not a recognised ''modeOfOperation'' value.'])
end   

max_val = max(abs([reference_sig(:); degraded_sig(:)]));

for m = 1:numel(mOp)
    pesqArgs = {['+' num2str(Fs)], ...
                         mOp{m}, ...
                         single(reference_sig / max_val), ...
                         single(degraded_sig / max_val)};
    res(m,:) = pesq_mex_(pesqArgs{~cellfun(@isempty,pesqArgs)});
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