%% Obsolete. Bayesian Optimization Doesn't work out this way...
G = FC6Generator("matlabGANfc6.mat");
net = alexnet;
%%
opt_z = optimvar('z',1,4096,'Type','continuous','LowerBound',-5,'UpperBound',5);
contrast = @(opt)std(G.visualize(opt.z),0,'all');
result = bayesopt(contrast, opt_z, 'AcquisitionFunctionName','expected-improvement-plus','IsObjectiveDeterministic',true,...
    'MaxObjectiveEvaluations',3000,'MaxTime',600);