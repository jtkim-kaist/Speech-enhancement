function RelFiles = buildReleaseZIP( WorkingDir, zipFileName, MainFile, RelFiles, RuntimeDependencies )
% Creates a ZIP file of all release dependencies for a main file
% 
% Syntax:	BUILDRELEASEZIP( WORKINGDIR, ZIPFILENAME, MAINFILE, RUNTIMEDEPENDENCIES )
% 
% Inputs: 
% 	WorkingDir          - The working directory of the project to release.
% 	zipFileName         - The name the final ZIP file.
% 	MainFile            - The path to the file for which to determine the
%                         dependencies and consequently build the release
%                         ZIP.
%   RelFiles            - A list of files to release (the main file's 
%                         dependencies are added to this list).
% 	RuntimeDependencies - If true, dependencies are found during a profiled
%                         runtime.
% 
% Outputs: 
% 	RelFiles            - The final list of released files.
% 
% Example: 
%     wrkdir = 'C:\myProject\';
%     zipFname = 'myProject_release';
%     mainFile = 'C:\myProject\examples\fullyfledgedexample.m';
%     buildReleaseZIP( wrkdir, zipFname, mainFile, {}, true )
% 
% See also: requiredFilesAndProducts, zip

% Author: Jacob Donley
% University of Wollongong
% Email: jrd089@uowmail.edu.au
% Copyright: Jacob Donley 2017
% Date: 09 August 2017
% Version: 1.1 (09 August 2017)
% Version: 1.0 (07 May 2017)
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Get all dependencies of the main file
flist = matlab.codetools.requiredFilesAndProducts( MainFile );                      % Full dependency file list

%% Get dependencies of the main file during runtime (includes anonymous functions and dynamic function handles)
% The MATLAB profiler is used here and can take quite some time if the main file is slow to run
if RuntimeDependencies
    profile on;                                                                        % Turn the profiler on
    [p,f]=fileparts(MainFile);
    func = strrep(strrep([p filesep f],filesep,'.'),'+','');                           % Deal with class folders and create function handle
    evalin('caller',[func ';']);                                                       % Assume file runs without arguments and run
    p = profile('info');                                                               % Stop the profiler after execution and get the profiler information
    flist = [flist {p.FunctionTable.FileName}];                                        % Append the runtime functions to the list
    dbs=dbstack;                                                                       % Get history of called functions for this runtime
    flist(contains(flist,{dbs.name})) = [];                                            % Remove references to this function and parents
end

%% Only keep files in working directory
RelFiles = unique([RelFiles(:)' flist(contains(flist,WorkingDir))]);                    % Release files

%% Copy to temp directory so zipping retains folder structure
newdirs = unique(cellfun(@fileparts, strrep(RelFiles, WorkingDir, ''),'un',0));     % Determine structure
tmpdir = [tempname filesep];                                                        % Get temp directory
cellfun(@(a) mkdir( [tmpdir a] ), newdirs);                                         % Create temp folder structure
cellfun(@(a,b) copyfile(a,[tmpdir b]), RelFiles, strrep(RelFiles, WorkingDir, '')); % Copy

%% Zip the entire folder
zip(zipFileName, tmpdir, WorkingDir);

%% Clean up
rmdir(tmpdir,'s');                                                                  % Delete all the temporarily copied files

end

