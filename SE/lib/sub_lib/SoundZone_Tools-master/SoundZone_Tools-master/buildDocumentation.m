function docFiles = buildDocumentation( WorkingDir, DocDir, MainFile, DocFiles, RuntimeDependencies, ThirdPartyHTML )
% Generates documentation HTML and builds MATLAB search database for dependencies of a main file
% 
% Syntax:	DOCFILES = BUILDDOCUMENTATION( WORKINGDIR, DOCDIR, MAINFILE, DOCFILES, RUNTIMEDEPENDENCIES, THIRDPARTYHTML )
% 
% Inputs: 
% 	WorkingDir          - The working directory of the project to document.
% 	zipFileName         - The directory for the documentation.
% 	MainFile            - The path to the file for which to determine the
%                         dependencies and consequently build the
%                         documentation.
%   DocFiles            - A list of files to document (the main file's 
%                         dependencies are added to this list).
% 	RuntimeDependencies - If true, dependencies are found during a profiled
%                         runtime.
%   ThirdPartyHTML      - Use third party HTML documentation generator
%                         called 'm2html'.
% 
% Outputs: 
% 	docFiles            - The final list of documented files.
% 
% Example: 
%     wrkdir = 'C:\myProject\';
%     docdir = 'doc';
%     mainFile = 'C:\myProject\examples\fullyfledgedexample.m';
%     buildDocumentation( wrkdir, docdir, mainFile, {}, true, false )
% 
% See also: publish, builddocsearchdb, doc, requiredFilesAndProducts

% Author: Jacob Donley
% University of Wollongong
% Email: jrd089@uowmail.edu.au
% Copyright: Jacob Donley 2017
% Date: 07 July 2017 
% Version: 0.1 (07 July 2017)
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin < 4, RuntimeDependencies = false; end

PubOpts ={...
    'format',               'html',             ...
    'stylesheet',           '',                 ...
    'createThumbnail',      true,               ...
    'figureSnapMethod',     'entireGUIWindow',  ...
    'imageFormat',          'png',              ...
    'maxHeight',            [],                 ...
    'maxWidth',             [],                 ...
    'useNewFigure',         true,               ...
    'evalCode',             false,              ...
    'catchError',           true,               ...
    ...'codeToEvaluate',       [],                 ...
    'maxOutputLines',       Inf,                ...
    'showCode',             true,               ...
    };

m2htmlOpts ={...
    'recursive',            'off',              ...
    'source',               'off',              ...
    'download',             'off',              ...
    'syntaxHighlighting',   'on',               ...
    'tabs',                 4,                  ...
    'globalHypertextLinks', 'off',              ...
    'todo',                 'on',               ...
    'graph',                'off',              ...
    'indexFile',            'index',            ...
    'extension',            '.html',            ...
    'template',             'blue',             ...
    'search',               'off',              ...
    'ignoredDir',           {'.svn' 'cvs'},     ...
    'save',                 'off',              ...
    'verbose',              'off',              ...
    };

%% Get all dependencies of the main file
flist = matlab.codetools.requiredFilesAndProducts( MainFile );                         % Full dependency file list

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
DocFiles = unique([DocFiles(:)' flist(contains(flist,WorkingDir))]);                       % Files to document
DocFiles = keepFilesInSubDirs(DocFiles,WorkingDir);                                    % Keep files only in subdirectories

%% Create documentation directory
docdirs = strrep(cellfun(@fileparts, strrep(DocFiles, WorkingDir, ''),'un',0),'+',''); % Determine structure
htmldir = [WorkingDir DocDir filesep 'html' filesep];                                  % Set html directory
if ~ThirdPartyHTML
    w = warning('query','MATLAB:MKDIR:DirectoryExists');                               % Get mkdir warning state
    warning('off','MATLAB:MKDIR:DirectoryExists');                                     % Turn off unnecessary warning
    cellfun(@(a) mkdir( [htmldir a] ), unique(docdirs));                               % Create html folder structure
    warning(w.state,'MATLAB:MKDIR:DirectoryExists');                                   % Reset warning state
end
%% Find indices of compatible files only
[~,~,e]=cellfun(@fileparts, DocFiles,'un',0);                                          % Get file extensions
Icode = cellfun(@(x) (strcmp(x,'.m') || strcmp(x,'.mlx')),e);                          % Determine if file is only code (determine if compatible for documentation)

%% Generate documentation
if ~ThirdPartyHTML
    cellfun(@(f,d) publish(f,PubOpts{:},'outputDir',[htmldir d]),DocFiles(Icode),docdirs(Icode),'un',0);
else
    m2html('mfiles',strrep(DocFiles(Icode),WorkingDir,''),'htmldir',htmldir,m2htmlOpts{:});
end
    
docFiles = DocFiles(Icode);
end

function fileList = keepFilesInSubDirs( fileList, workingDir )

fileList( ...
    cellfun( ...
    @isempty,...
    strrep( ...
    cellfun(@fileparts,fileList,'un',0),...
    fileparts(workingDir),'')  ) )...
    = [];

end
