%% |dirPlus| demo
% The |dirPlus| function recursively collects a list of files or
% subdirectories from a folder tree, allowing you to specify the selection
% criteria for how the list is collected and formatted. For the following
% examples, we'll be using the main MATLAB toolbox path.
%%

rootPath = 'C:\Program Files\MATLAB\R2016b\toolbox\matlab';
format compact

%% The 'FileFilter' option
% By default, |dirPlus| will collect a list of files. We can specify a
% <https://www.mathworks.com/help/matlab/matlab_prog/regular-expressions.html
% regular-expression> pattern to filter the file names on, collecting a
% list of files that match. Here's an example that uses the 'FileFilter'
% option to recursively find every file with a '.m' extension:

fileList = dirPlus(rootPath, 'FileFilter', '\.m$');
fprintf('%d files found.\n', size(fileList, 1));
fprintf('%s\n', fileList{1:5}, '...');

%%
% It's a pretty long list, so I've only shown the first 5 files it finds.
% Notice they are listed with the full path prepended by default.

%%
% If you have multiple match expressions to filter on, you can use
% <https://www.mathworks.com/help/matlab/matlab_prog/regular-expressions.html#f0-42884
% grouping operators> to include them all in one expression. For example,
% this will find every '.jpg', '.bmp', and '.tif' file:

fileList = dirPlus(rootPath, 'FileFilter', '\.(jpg|bmp|tif)$');
fprintf('%d files found.\n', size(fileList, 1));
fprintf('%s\n', fileList{1:5}, '...');

%% The 'ReturnDirs' option
% We can instead collect a list of subdirectories by setting the
% 'ReturnDirs' option to |true|:

dirList = dirPlus(rootPath, 'ReturnDirs', true);
fprintf('%d directories found.\n', size(dirList, 1));
fprintf('%s\n', dirList{1:5}, '...');

%%
% Note that the output list will be ordered by subdirectory depth (i.e. all
% subdirectories at depth 0, followed by all subdirectories at depth 1,
% etc.). Any settings for 'FileFilter' will be ignored.

%% The 'DirFilter' option
% We can filter subdirectories just like we filter files using the
% 'DirFilter' option. This will give us a list of subdirectories containing
% the string |'addons'|:

dirList = dirPlus(rootPath, 'ReturnDirs', true, 'DirFilter', 'addons');
fprintf('%d directories found.\n', size(dirList, 1));
fprintf('%s\n', dirList{:});

%%
% There are 4 subdirectories in our root directory with |'addons'| in the
% name. Let's say we wanted to find any '.m' files in subdirectories with
% |'addons'| in the name. We can use the 'DirFilter' and 'FileFilter'
% options like so:

fileList = dirPlus(rootPath, 'DirFilter', 'addons', 'FileFilter', '\.m$');
fprintf('%d files found.\n', size(fileList, 1));
fprintf('%s\n', fileList{:});

%%
% No files found? We know from the <dirPlus_demo.html#2 *'FileFilter'*>
% example above that there are '.m' files present in
% |rootPath\addons\cef\+matlab\+internal\+addons\|. Shouldn't these show
% up? And for that matter, why didn't '+addons' show up in the list of
% subdirectories containing the string |'addons'|? There's another option
% setting at work here...

%% The 'RecurseInvalid' option
% When a subdirectory doesn't match the 'DirFilter' regular expression
% pattern (marking it as "invalid"), a few things happen:
%
% # That subdirectory is not included in any output lists returned with the
% 'ReturnDirs' option.
% # The file contents of that subdirectory are not included in any output
% lists returned.
% # *By default*, the recursive search does not go any further down the
% folder tree for that subdirectory. This is why the '+addons' folder is
% not found or searched through in the prior examples: it is nested within
% invalid folders (namely, 'cef').
%
% In some cases we'd like the recursive search to continue through
% subdirectories even when their contents are excluded. Setting the
% 'RecurseInvalid' option to |true| accomplishes this:

fileList = dirPlus(rootPath, 'DirFilter', 'addons', 'FileFilter', '\.m$', ...
                             'RecurseInvalid', true);
fprintf('%d files found.\n', size(fileList, 1));
fprintf('%s\n', fileList{:});

%%
% And now we can see those '.m' files in the deeply nested '+addons'
% subdirectory.

%% The 'Struct' option
% Instead of getting the output as a cell array of files/subdirectories, we
% can have |dirPlus| return a structure array of the form returned by the
% <https://www.mathworks.com/help/matlab/ref/dir.html |dir|> function by
% setting the 'Struct' option to |true|:

fileList = dirPlus(rootPath, 'Struct', true, 'FileFilter', '\.m$');
fprintf('%d files found.\n', size(fileList, 1));
display(fileList(1));

%% The 'Depth' option
% If we don't want to search quite so far down the folder tree, we can
% limit the search depth with the 'Depth' option. Let's see how many '.m'
% files are in the root folder:

fileList = dirPlus(rootPath, 'FileFilter', '\.m$', 'Depth', 0);
fprintf('%d files found.\n', size(fileList, 1));

%%
% Looks like none are. They are all contained in subdirectories. Let's see
% how many are located in just the immediate subdirectories of the root
% folder:

fileList = dirPlus(rootPath, 'FileFilter', '\.m$', 'Depth', 1);
fprintf('%d files found.\n', size(fileList, 1));

%% The 'PrependPath' option
% Maybe we just want the file/subdirectory names, but don't care about the
% absolute paths. In this case, we just set the 'PrependPath' option to
% |false|:

fileList = dirPlus(rootPath, 'FileFilter', '\.m$', 'PrependPath', false);
fprintf('%d files found.\n', size(fileList, 1));
fprintf('%s\n', fileList{1:5}, '...');

%% The 'ValidateFileFcn' option
% Sometimes we might want to select files based on a more complicated
% criteria than just what's in their names. In this case, we can use the
% 'ValidateFileFcn' option to specify a function that is to be run on each
% file found. This function should accept as input a structure of the form
% returned by the <https://www.mathworks.com/help/matlab/ref/dir.html
% |dir|> function and return a logical value (|true| to collect it in the
% list, |false| to ignore it). First, let's find all the '.png' files:

fileList = dirPlus(rootPath, 'FileFilter', '\.png$');
fprintf('%d files found.\n', size(fileList, 1));

%%
% Now, we can specify an anonymous function that gets the byte size of each
% file and returns |true| for only those greater than 250KB:

bigFcn = @(s) (s.bytes > 512^2);
fileList = dirPlus(rootPath, 'FileFilter', '\.png$', ...
                             'ValidateFileFcn', bigFcn);
fprintf('%s\n', fileList{:});

%%
% Just the one.

%% The 'ValidateDirFcn' option
% We can apply more complicated validation criteria for subdirectories as
% well using the 'ValidateDirFcn' option. Let's say we want to find all the
% '.m' files that are not contained within a
% <https://www.mathworks.com/help/matlab/matlab_oop/scoping-classes-with-packages.html
% package folder> (i.e. one that starts with '+'). Here's one way to do it:

dirFcn = @(s) ~strcmp(s.name(1), '+');
fileList = dirPlus(rootPath, 'ValidateDirFcn', dirFcn, 'FileFilter', '\.m$');
fprintf('%d files found.\n', size(fileList, 1));
fprintf('%s\n', fileList{1:5}, '...');

%%
% In this case, you could actually do this using a 'DirFilter' regular
% expression as well:

fileList = dirPlus(rootPath, 'DirFilter', '^[^+]', 'FileFilter', '\.m$');
fprintf('%d files found.\n', size(fileList, 1));
fprintf('%s\n', fileList{1:5}, '...');