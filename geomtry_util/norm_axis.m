function norms = norm_axis(Vecs, dim)
norms = sqrt(sum(Vecs.^2,dim));
end
