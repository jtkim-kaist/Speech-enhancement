function [Files] = keepFilesFromFolder( FileList, KeepFromFolder )
% Keeps files and file paths in a cell array if the file names in a given folder are found in the path string
% 
% Syntax:	[Files] = KEEPFILESFROMFOLDER(FileList,KeepFromFolder) 
% 
% Inputs: 
% 	FileList       - A list of files to filter
% 	KeepFromFolder - A folder containing filenames from which to search in
%                    the given FileList and keep.
% 
% Outputs: 
% 	Files - List of files with only the file names found in the given
% 	folder still in the list
% 
% 
% See also: getAllFiles

% Author: Jacob Donley
% University of Wollongong
% Email: jrd089@uowmail.edu.au
% Copyright: Jacob Donley 2017
% Date: 18 January 2017 
% Revision: 0.1
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Just incase this function tries to call within a class folder we
% should create a function handle for other functions in that class folder
inf = dbstack('-completenames');
funcName = 'getAllFiles';
funcPath = inf.file;
classDirs = getClassDirs(funcPath);
getAllFilesFn = str2func([classDirs funcName]);


%% KEEPFILESFROMFOLDER
    [~,fnames]=cellfun(@fileparts, getAllFilesFn( KeepFromFolder ),'UniformOutput',false);
    for i=1:numel(fnames)
        tmp = strfind(FileList,fnames{i});
        I = arrayfun(@(x) ~isempty(tmp{x}),1:numel(FileList));
        fileKeeps(I) = FileList(I);
    end
    fileKeeps(cellfun('isempty',fileKeeps))=[];
    Files = {fileKeeps{:}}.';


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