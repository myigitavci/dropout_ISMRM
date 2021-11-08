function fp = listfile(str, cellflag)
[path, name, ext] = fileparts2(str);
fp = dir(str);
if ~exist('cellflag', 'var')
    if length(fp) == 1 
        fp = fullfile(path, fp.name);
    else
        fp = {fp(:).name}';
    end
else
    fp = {fp(:).name}';
end