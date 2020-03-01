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
fullstr = jsonencode(options); % use jsonencode as a proxy for text representation of options
fullstr = strrep(fullstr, ':', sprintf(': ')); % increase space after :
fullstr = strrep(fullstr, ',', sprintf(',\r')); % change line after each , 
fullstr = strrep(fullstr, '_', sprintf(' ')); % avoid `_` interpreted as Latex
fullstr = strrep(fullstr, '{', sprintf(''));
fullstr = strrep(fullstr, '}', sprintf(''));
end