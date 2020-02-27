%% Step Size Expectation 
% Script to compute the statistical expectation of the step size 
dimen = 4096; 
pop_size = 40;
weights = rankweight(pop_size,pop_size/2);
mu = 0.5;
basis_norm = 100;

basis_vec = zeros(1, dimen);
basis_vec(1) = basis_norm; 
tang_codes = mu * randn(pop_size, dimen);


function 