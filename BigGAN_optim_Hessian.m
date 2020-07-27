%% BigGAN_optim_Hessian

global G net eigvect eigvals
net = alexnet;
G = torchBigGAN();
my_final_path = "E:\OneDrive - Washington University in St. Louis\HessTune\HessOptim_BigGAN";
%%
py.importlib.import_module("numpy")
% data = py.numpy.load("E:\OneDrive - Washington University in St. Louis\HessTune\NullSpace\Evolution_Avg_Hess.npz");
data = py.numpy.load("E:\Cluster_Backup\BigGANH\summary\H_avg_1000cls.npz");
% eigvect = data.get("eigvect_avg").double; % each column is an eigen vector. 
% eigvals = data.get("eigv_avg").double;
% eigvals = eigvals(end:-1:1);
% eigvect = eigvect(:,end:-1:1);
eigvals = data.get("eigvals_clas_avg").double; % each column is an eigen vector. 
eigvect = data.get("eigvects_clas_avg").double;
eigvals = eigvals(end:-1:1);
eigvect = eigvect(:,end:-1:1);
%%
data = py.numpy.load("E:\OneDrive - Washington University in St. Louis\ref_img_fit\Pasupathy\Nullspace\Pasu_Space_Avg_Hess.npz");
eigvect_fc6 = data.get("eigvect_avg").double; % each column is an eigen vector. 
eigvals_fc6 = data.get("eigv_avg").double;
eigvect_fc6 = eigvect_fc6(:,end:-1:1);
eigvals_fc6 = eigvals_fc6(end:-1:1);
%%
G = torchBigGAN();
%%
G = G.select_space("class");
Optimizer = CMAES_simple(128, [], struct("init_sigma",0.06));
n_gen = 100;unit = {"fc6",2};param_lab="CMA_ctrl";fign=[];nsr=0;init_genes = 0.04 * randn(1,128);
[scores_all,codes_all,generations,norm_all,Optimizer,h,exp_id,exp_dir] = GANoptim_test_fun(unit, ...
    n_gen, Optimizer, init_genes, nsr, fign, param_lab, my_final_path);
%
Optimizer = CMAES_Hessian(128, 128, [], struct("init_sigma",0.12));
Optimizer.getHessian(eigvect, eigvals, 128, "1/3"); 
n_gen = 100;unit = {"fc6",2};param_lab="CMA_Hess1_3";fign=[];nsr=0;init_genes = 0.04 * randn(1,128);
[scores_all,codes_all,generations,norm_all,Optimizer,h,exp_id,exp_dir] = GANoptim_test_fun(unit, ...
    n_gen, Optimizer, init_genes, nsr, fign, param_lab, my_final_path);
%%
G = FC6Generator();
Optimizer = CMAES_simple(4096, [], struct("init_sigma",3));
n_gen = 100;unit = {"fc6",2};param_lab="CMA_ctrl_fc6";fign=[];nsr=0;init_genes = 1 * randn(1,4096);
[scores_all,codes_all,generations,norm_all,Optimizer,h,exp_id,exp_dir] = GANoptim_test_fun(unit, ...
    n_gen, Optimizer, init_genes, nsr, fign, param_lab, my_final_path);
% 
Optimizer = CMAES_Hessian(4096, 800, [], struct("init_sigma",3.5));
Optimizer.getHessian(eigvect_fc6, eigvals_fc6, 800, "1/3");
n_gen = 100;unit = {"fc6",2};param_lab="Hess800_cub35";fign=[];nsr=0;init_genes = 1 * randn(1,4096);
[scores_all,codes_all,generations,norm_all,Optimizer,h,exp_id,exp_dir] = GANoptim_test_fun(unit, ...
    n_gen, Optimizer, init_genes, nsr, fign, param_lab, my_final_path);

%%
% Optimizer = CMAES_Hessian(4096, 800, [], struct("init_sigma",7));
% Optimizer.getHessian(eigvect_fc6, eigvals_fc6, 800, "1/3");
% n_gen = 100;unit = {"fc6",2};param_lab="Hess800_sqrt7";fign=[];nsr=0;init_genes = 1 * randn(1,4096);
% [scores_all,codes_all,generations,norm_all,Optimizer,h,exp_id,exp_dir] = GANoptim_test_fun(unit, ...
%     n_gen, Optimizer, init_genes, nsr, fign, param_lab, my_final_path);
%%
tic
G = torchBigGAN();
G = G.select_space("class");
unit_list = {{"fc6",5}, {"fc7",5}, {"fc8",5}, {"conv5",5,[7,7]}, {"conv2",5,[14,14]}};
scores_col = cell(5, 2, 5);
scores_med_col = cell(5, 2, 5);
for uniti = 1:numel(unit_list)
unit = unit_list{uniti};
for triali = 1:5
init_genes = 0.04 * randn(1,128);
%%
Optimizer = CMAES_simple(128, [], struct("init_sigma",0.06));
param_lab="CMA_ctrl";fign=[];nsr=0;
[scores_all,codes_all,generations,norm_all,Optimizer,h,exp_id,exp_dir] = GANoptim_test_fun(unit, ...
    n_gen, Optimizer, init_genes, nsr, fign, param_lab, my_final_path);
scores_col{uniti, 1, triali} = scores_all(generations==100);
scores_med_col{uniti, 1, triali} = scores_all(generations==50);
%%
Optimizer = CMAES_Hessian(128, 128, [], struct("init_sigma",0.20));
Optimizer.getHessian(eigvect, eigvals, 128, "1/5"); % 
param_lab="CMA_Hess1_5";fign=[];nsr=0;
[scores_all,codes_all,generations,norm_all,Optimizer,h,exp_id,exp_dir] = GANoptim_test_fun(unit, ...
    n_gen, Optimizer, init_genes, nsr, fign, param_lab, my_final_path);
scores_col{uniti, 2, triali} = scores_all(generations==100);
scores_med_col{uniti, 2, triali} = scores_all(generations==50);
toc
end
end
%%
outdir = "E:\OneDrive - Washington University in St. Louis\HessTune\HessOptim_BigGAN";
save(fullfile(outdir, "scores_summary.mat"), 'scores_col', 'scores_med_col','unit_list')
%%
% scores_col{uniti, 1, triali}
% scores_col{uniti, 2, triali}
scores_arr = cellfun(@mean, scores_col);
target_str = cellfun(@(unit)compose("%s-%d",unit{1},unit{2}), unit_list);
figure(1);clf;hold on 
for i = 1:5
jitter = randn(5,1)*0.1;
scatter(i+jitter,squeeze(scores_arr(i,1,:)),50,'blue')
scatter(0.3+i+jitter,squeeze(scores_arr(i,2,:)),50, 'red')
line([i, 0.3+i]'+jitter',squeeze(scores_arr(i,:,:)), 'color', 'k')
end
xticks([1:5]+0.15)
xticklabels(target_str)
xlabel("Layer-channel")
ylabel("raw activation")
legend({"CMAES","CMAES-Hessian"},'location','best')
title(["Comparison of CMAES and CMAES-Hessian method","mean 100 gen","CMAES (sigma=0.06) CMAES hess (sigma=0.20, 1/5 scaling)"])
saveas(1,fullfile(outdir,"CMAES-Hess-100gen_mean.jpg"))
%%
scores_arr = cellfun(@max, scores_col);
target_str = cellfun(@(unit)compose("%s-%d",unit{1},unit{2}), unit_list);
figure(1);clf;hold on 
for i = 1:5
jitter = randn(5,1)*0.1;
scatter(i+jitter,squeeze(scores_arr(i,1,:)),50,'blue')
scatter(0.3+i+jitter,squeeze(scores_arr(i,2,:)),50, 'red')
line([i, 0.3+i]'+jitter',squeeze(scores_arr(i,:,:)), 'color', 'k')
end
xticks([1:5]+0.15)
xticklabels(target_str)
xlabel("Layer-channel")
ylabel("raw activation")
legend({"CMAES","CMAES-Hessian"},'location','best')
title(["Comparison of CMAES and CMAES-Hessian method","max 100 gen","CMAES (sigma=0.06) CMAES hess (sigma=0.20, 1/5 scaling)"])
saveas(1,fullfile(outdir,"CMAES-Hess-100gen_max.jpg"))
%%
scores_arr = cellfun(@mean, scores_med_col);
target_str = cellfun(@(unit)compose("%s-%d",unit{1},unit{2}), unit_list);
figure(1);clf;hold on 
for i = 1:5
jitter = randn(5,1)*0.1;
scatter(i+jitter,squeeze(scores_arr(i,1,:)),50,'blue')
scatter(0.3+i+jitter,squeeze(scores_arr(i,2,:)),50, 'red')
line([i, 0.3+i]'+jitter',squeeze(scores_arr(i,:,:)), 'color', 'k')
end
xticks([1:5]+0.15)
xticklabels(target_str)
xlabel("Layer-channel")
ylabel("raw activation")
legend({"CMAES","CMAES-Hessian"},'location','best')
title(["Comparison of CMAES and CMAES-Hessian method","mean 50 gen","CMAES (sigma=0.06) CMAES hess (sigma=0.20, 1/5 scaling)"])
saveas(1,fullfile(outdir,"CMAES-Hess-50gen_mean.jpg"))
%%
scores_arr = cellfun(@max, scores_med_col);
target_str = cellfun(@(unit)compose("%s-%d",unit{1},unit{2}), unit_list);
figure(1);clf;hold on 
for i = 1:5
jitter = randn(5,1)*0.1;
scatter(i+jitter,squeeze(scores_arr(i,1,:)), 50, 'blue')
scatter(0.3+i+jitter,squeeze(scores_arr(i,2,:)), 50, 'red')
line([i, 0.3+i]'+jitter',squeeze(scores_arr(i,:,:)), 'color', 'k')
end
xticks([1:5]+0.15)
xticklabels(target_str)
xlabel("Layer-channel")
ylabel("raw activation")
legend({"CMAES","CMAES-Hessian"},'location','best')
title(["Comparison of CMAES and CMAES-Hessian method","max 50 gen","CMAES (sigma=0.06) CMAES hess (sigma=0.20, 1/5 scaling)"])
saveas(1,fullfile(outdir,"CMAES-Hess-50gen_max.jpg"))
%%
