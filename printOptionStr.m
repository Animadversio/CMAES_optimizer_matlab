function fullstr = printOptionStr(options)
% Fields = fieldnames(options);
% opt_strs = {};
% fullstr = "";
% for Fi = 1:length(Fields)
%     val = getfield(options, Fields{Fi});
%     opt_strs{Fi} = sprintf("%s=%s\n",Fields{Fi},num2str(val));
%     fullstr = strcat(fullstr, opt_strs{Fi});
% end
% https://www.mathworks.com/matlabcentral/answers/335296-format-output-from-jsonencode-to-make-it-more-human-readable
fullstr = jsonencode(options);
fullstr = strrep(fullstr, ',', sprintf(',\r'));
fullstr = strrep(fullstr, '{', sprintf(''));
fullstr = strrep(fullstr, '}', sprintf(''));
end