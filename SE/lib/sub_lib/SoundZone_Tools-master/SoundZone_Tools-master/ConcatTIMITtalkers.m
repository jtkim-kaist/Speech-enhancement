function ConcatTIMITtalkers( TIMITdir, OutDir )
% Concatenates all the talkers from the TIMIT corpus into individual speech files 
% 
% Syntax:	CONCATTIMITTALKERS( TIMITDIR, OUTDIR )
% 
% Inputs: 
% 	TIMITdir - The directory of the TIMIT corpus
% 	OutDir - The output directory to save the concatenated speech files
% 
% Example: 
% 	TIMITtlkrscat('C:\TIMIT_90\', '.\')
% 
% See also: getAllFiles, audioread, audiowrite

% Author: Jacob Donley
% University of Wollongong
% Email: jrd089@uowmail.edu.au
% Copyright: Jacob Donley 2017
% Date: 21 April 2017 
% Revision: 0.1
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
AudioExt = '.wav';

% Get all files in the TIMIT corpus
AllFiles = getAllFiles( TIMITdir );

% Limit the list to only the WAV files and sort them
WAVfiles = sort( AllFiles( contains(lower(AllFiles), AudioExt) ) );

% Reduce to just the unique talker directories
TalkerDirs = unique(cellfun(@fileparts,WAVfiles,'un',0));

% Create the output directories if they don't exist
OutDirs = strrep(TalkerDirs, TIMITdir, [fileparts(OutDir) filesep]);
cellfun(@newDir,cellfun(@fileparts,OutDirs,'un',0),'un',0);

% Partition the speech into the separate talkers
PartDirs = cellfun(@(tds) WAVfiles(contains(WAVfiles,tds)),TalkerDirs,'un',0);

% Concatenate all the speech files
[Y, FS] = cellfun(@ReadCatAudio, PartDirs, 'un', 0);

% Write all of the concatenated speech files to disk
arrayfun(@(i) audiowrite([OutDirs{i} AudioExt], Y{i}, FS{i}), 1:numel(Y));

end

function [y, Fs] = ReadCatAudio( filenames )
[Y, FS] = cellfun(@audioread,filenames,'un',0);
if ~isequal(FS{:}),error('Different sampling frequencies detected.');end
Y_=cellfun(@transpose,Y,'un',0);
y = [Y_{:}]; Fs = FS{1};
end

function newDir(dir)
  narginchk(1,1);
  if ~exist(dir,'dir'), mkdir(dir); end
end
