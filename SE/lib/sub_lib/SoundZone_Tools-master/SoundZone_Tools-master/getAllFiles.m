function fileList = getAllFiles(dirPath)
% Retrieves a list of all files within a directory
% 
% Syntax:	fileList = getAllFiles(dirName)
% 
% Inputs: 
%       dirPath - The relative or full path of the directory to recursivley
%       search.
%
% Outputs: 
%       fileList - A cell array list of the full path for each file found.
%
% Example: 
%       searchPath = [matlabroot filesep 'examples'];
%       files = getAllFiles(searchPath);
%
% 
% Author: Jacob Donley
% University of Wollongong
% Email: jrd089@uowmail.edu.au
% Date: 22 January 2015
% Revision: 0.1
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Just incase this function tries to recursively call within a class folder we
% should create a function handle for this function to use
infun = dbstack('-completenames');
funcName = infun.name;
funcPath = infun.file;
classDirs = getClassDirs(funcPath);
thisFuncHandle = str2func([classDirs funcName]);

% Find the files
  dirData = dir(dirPath);      % Get the data for the current directory
  dirIndex = [dirData.isdir];  % Find the index for directories
  fileList = {dirData(~dirIndex).name}';  %'# Get a list of the files
  if ~isempty(fileList)
    fileList = cellfun(@(x) fullfile(dirPath,x),...  % Prepend path to files
                       fileList,'UniformOutput',false);
  end
  subDirs = {dirData(dirIndex).name};  % Get a list of the subdirectories
  validIndex = ~ismember(subDirs,{'.','..'});  % Find index of subdirectories that are not '.' or '..'
  for iDir = find(validIndex)                  % Loop over valid subdirectories
    nextDir = fullfile(dirPath,subDirs{iDir});    % Get the subdirectory path
    fileList = [fileList; thisFuncHandle(nextDir)];  % Recursively call getAllFiles
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