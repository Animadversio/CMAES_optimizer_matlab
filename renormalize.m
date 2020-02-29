function codes_renorm = renormalize(codes, norms)
    assert(size(codes,1) == length(norms) || length(norms) == 1)
    norms = reshape(norms, [],1);
    code_norm = sqrt(sum(codes.^2, 2));
    codes_renorm = norms .* codes ./ code_norm;  % norms should be a 1d array
return 