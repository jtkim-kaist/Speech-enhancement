function [ spect, frqs ] = LTASS( speech_folder_OR_vec, nfft, fs )
% Computes the Long-Term Average Speech Spectrum from a folder of speech files or vector of speech samples
%
% Syntax:	[ spect, frqs ] = LTASS( speech_folder_OR_vec, nfft )
%
% Inputs:
% 	speech_folder_OR_vec - The path to the folder containing the speech
%                          files OR a vector of concatenated speech signals
% 	nfft - The number of FFT points used to compute the LTASS
% 	fs - The sampling frequency to use (if not provided then the sampling
%        frequency of the file is used)
%
% Outputs:
% 	spect - The LTASS spectrum
% 	frqs - The frequency vector for the spectrum

% Author: Jacob Donley
% University of Wollongong
% Email: jrd089@uowmail.edu.au
% Copyright: Jacob Donley 2017
% Date: 17 June 2016
% Revision: 0.4 (30 March 2017)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if isa(speech_folder_OR_vec,'char') % if a character array (string)
    % Just incase this function tries to call getAllFiles within a class folder we
    % should create a function handle for getAllFiles regardless
    inf = dbstack('-completenames');
    funcName = 'getAllFiles';
    funcPath = inf.file;
    classDirs = getClassDirs(funcPath);
    getAllFiles_ = str2func([classDirs funcName]);
    
    %% Start LTASS
    files = getAllFiles_(speech_folder_OR_vec);
    speech=[];
    F = length(files);
    for file = 1:F
        try
            [audioSig,fs_] = audioread(files{file});
            if nargin < 3, fs = fs_; end
            audioSig = audioSig ./ rms(audioSig(:));
        catch err
            if strcmp(err.identifier, 'MATLAB:audiovideo:audioread:FileTypeNotSupported')
                continue; % Skip unsupported files
            end
        end
        speech = [speech; audioSig];
    end
    if nargin < 2
        nfft = numel(speech);
    end
    if logical(mod(nfft,2)) % if isodd( nfft )
        nfft = nfft-1; % Force nfft to be even so that pwelch returns normalised frequencies [0,...,1]
    end

else
    speech = speech_folder_OR_vec;
end

%%
win_=rectwin(nfft);
ovlap = 0;

[spect,frqs]=pwelch(speech,win_,nfft*ovlap,nfft,fs,'power'); % Power spectrum
spect = sqrt(spect); % Magnitude spectrum

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