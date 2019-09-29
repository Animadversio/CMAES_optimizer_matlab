A = [0.9810    0.0172    0.0585;
    0.0203    0.9951    0.0191;
    0.0313   -0.0112    1.0079];
Ainv= inv(A);
c1 = 0.1026; 
pc = [0,0,1];
%%
v = pc * Ainv;
normv = v * v';
A = sqrt(1-c1) * A + sqrt(1-c1)/normv*(sqrt(1 + normv *c1/(1-c1))-1) * v' * pc;
Ainv = 1/sqrt(1-c1) * Ainv - 1/sqrt(1-c1)/normv*(1-1/sqrt(1 + normv *c1/(1-c1))) * Ainv* (v'*v);
disp(A*Ainv)
disp(Ainv*A)