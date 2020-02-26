function y = ExpMap(x, tang_vec)
	% Modified Feb. 25th to allow x of arbitrary norm.
    EPS = 1E-3;
    assert(abs(norm(x)-1) < EPS);
    assert(sum(x * tang_vec') < EPS);
    angle_dist = sqrt(sum(tang_vec.^2, 2)); % vectorized
    uni_tang_vec = tang_vec ./ angle_dist;
    x = repmat(x, size(tang_vec, 1), 1); % vectorized
    y = cos(angle_dist) .* x + sin(angle_dist) .* uni_tang_vec;
end