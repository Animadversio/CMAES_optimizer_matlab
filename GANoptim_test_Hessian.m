%% Test Hessian 
 
global G net eigvect eigvals
net = alexnet;
G = FC6Generator;
%% Load the Avg Hessian Matrix 
py.importlib.import_module("numpy")
% data = py.numpy.load("E:\OneDrive - Washington University in St. Louis\HessTune\NullSpace\Evolution_Avg_Hess.npz");
data = py.numpy.load("E:\OneDrive - Washington University in St. Louis\ref_img_fit\Pasupathy\Nullspace\Pasu_Space_Avg_Hess.npz");
eigvect = data.get("eigvect_avg").double; % each column is an eigen vector. 
eigvals = data.get("eigv_avg").double;
eigvals = eigvals(end:-1:1);
eigvect = eigvect(:,end:-1:1);
%%
Optimizer = CMAES_Hessian(4096, 800, [], struct("init_sigma",7));
Optimizer.getHessian(eigvect, eigvals, 800, "1/3");
%
n_gen = 100;unit = {"fc8",2};param_lab="Hess800_sqrt7";fign=[];nsr=0;init_genes = zeros(1,4096);my_final_path = "E:\OneDrive - Washington University in St. Louis\HessTune\HessOptim";
[scores_all,codes_all,generations,norm_all,Optimizer,h,exp_id,exp_dir] = GANoptim_test_fun(unit, ...
    n_gen, Optimizer, init_genes, nsr, fign, param_lab, my_final_path);

%% BO optimization of hyper parameters
sigma = optimizableVariable('sigma',[1,20],'Type','real');
spect = optimizableVariable('spect',[200,1000],'Type','integer');
mode = optimizableVariable('mode',{'1/2','1/3','1/4'},'Type','categorical');
results = bayesopt(@optim_result,[sigma, spect, mode],'Verbose',1,...
    'AcquisitionFunctionName','lower-confidence-bound','MaxObjectiveEvaluations',50); % sigma   3.6235  spect 486   mode  1/4 
%% BO optimization of hyper parameters 200 generation evol
sigma = optimizableVariable('sigma',[1,20],'Type','real');
spect = optimizableVariable('spect',[200,1000],'Type','integer');
mode = optimizableVariable('mode',{'1/2','1/3','1/4'},'Type','categorical');
results200 = bayesopt(@optim_result,[sigma, spect, mode],'Verbose',1,...
    'AcquisitionFunctionName','lower-confidence-bound','MaxObjectiveEvaluations',100); % sigma   3.6235  spect 486   mode  1/4 
%%
optim_result(results.bestPoint)
%%
optim_result(struct("sigma",7,'mode',"1/3","spect",800))
%%
save("E:\OneDrive - Washington University in St. Louis\HessTune\HessOptim\BO_results.mat",'results','results200')
%% 100 generation 
% Optimization completed.
% MaxObjectiveEvaluations of 50 reached.
% Total function evaluations: 50
% Total elapsed time: 2627.8557 seconds.
% Total objective function evaluation time: 2586.8568
% Best observed feasible point:
%     sigma     spect    mode
%     ______    _____    ____
%     3.5157     477     1/4 
% Observed objective function value = -75.3291
% Estimated objective function value = -65.0252
% Function evaluation time = 50.069
% Best estimated feasible point (according to models):
%     sigma     spect    mode
%     ______    _____    ____
%     3.6235     486     1/4 
% Estimated objective function value = -65.0252
% Estimated function evaluation time = 51.4445
%% 200 generation optimization
% Optimization completed.
% MaxObjectiveEvaluations of 100 reached.
% Total function evaluations: 100
% Total elapsed time: 11603.7794 seconds.
% Total objective function evaluation time: 11508.4605
% 
% Best observed feasible point:
%     sigma     spect    mode
%     ______    _____    ____
%     3.5514     567     1/3 
% Observed objective function value = -103.9877
% Estimated objective function value = -93.2878
% Function evaluation time = 111.2443
% Best estimated feasible point (according to models):
%     sigma     spect    mode
%     ______    _____    ____
%     3.5998     549     1/3 
% Estimated objective function value = -93.2878
% Estimated function evaluation time = 113.3726
optim_result(results200.bestPoint)
function optim_val = optim_result(x)
global eigvect eigvals
Optimizer = CMAES_Hessian(4096, x.spect, [], struct("init_sigma",x.sigma));
Optimizer.getHessian(eigvect, eigvals, x.spect, x.mode);
n_gen = 200;unit = {"fc8",2};param_lab=compose("Hess%d_%s_%d",x.spect,x.mode,x.sigma);
fign=[];nsr=0;init_genes = zeros(1,4096);my_final_path = "E:\OneDrive - Washington University in St. Louis\HessTune\HessOptim";
[scores_all,codes_all,generations,~,~,~,~,~] = GANoptim_test_fun(unit, ...
    n_gen, Optimizer, init_genes, nsr, fign, param_lab, my_final_path);
optim_val = - mean(arrayfun(@(geni)max(scores_all(generations == geni)), n_gen-4:n_gen));
end