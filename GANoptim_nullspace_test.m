%%
npypath = "E:\OneDrive - Washington University in St. Louis\ref_img_fit\Pasupathy\Nullspace\Pasu_Space_Avg_Hess.npz";
py.importlib.import_module("numpy");
data = py.numpy.load(npypath); % eigv_avg  eigvect_avg  H_avg
eigvect_avg = data.get("eigvect_avg").double; % each column is a eigen vector. 
eigval_avg = data.get("eigv_avg").double;
H_avg = data.get("H_avg").double;
%%
global G net
net = alexnet;
G = FC6Generator('matlabGANfc6.mat');
addpath E:\Github_Projects\Monkey_Visual_Experiment_Data_Processing\utils\
addpath D:\Github\Monkey_Visual_Experiment_Data_Processing\utils\
addpath geomtry_util\
%%
savedir = "E:\OneDrive - Washington University in St. Louis\HessTune\NullSpace\Optim";
init_genes=randn(30,4096);nsr=0;
bn=400;
optimizer = CMAES_ReducDim(4096, [], bn, struct("init_sigma", 4, "Aupdate_freq",200));
optimizer.getBasis(eigvect_avg(:,2000-bn+1:2000));
unit = {"fc8",2};
[scores_all,codes_all,generations,norm_all,Optimizer,h,exp_id] = GANoptim_test_fun(unit, 100, optimizer, ...
    init_genes, nsr, [], "nullspace", savedir);
%%
savedir = "E:\OneDrive - Washington University in St. Louis\HessTune\NullSpace\Optim";
init_genes=randn(30,4096);nsr=0;
bn=100;
optimizer = CMAES_ReducDim(4096, [], bn, struct("init_sigma", 2, "Aupdate_freq",200));
% optimizer.getBasis(eigvect_avg(:,end-100-bn+1:end-100));
optimizer.getBasis("rand");
unit = {"fc8",2};
[scores_all,codes_all,generations,norm_all,Optimizer,h,exp_id] = GANoptim_test_fun(unit, 100, optimizer, ...
    init_genes, nsr, [], "nullspace", savedir);
%%
% unit_list = {{"fc8",2}, {"fc7",2}, {"fc6",2}, ...
%     {"conv5",2,[7,7]}, {"conv4",2,[7,7]}, {"conv3",2,[7,7]}, {"conv2",2,[14,14]}, {"conv1",2,[28,28]}};
h = figure();
unit_list = {{"fc8",2}, {"conv5",2,[7,7]}, {"conv2",2,[14,14]}};
savedir = "E:\OneDrive - Washington University in St. Louis\HessTune\NullSpace\Optim";
init_genes=randn(30,4096);nsr=0;
for uniti = 1:numel(unit_list)
unit = unit_list{uniti};
tic
for triali = 1:4
% sample certain number of dimensions at different position of the Hessian
% Spectrum
for bn=[50, 100, 200, 400]
for ofs = [0, 100, 500, 1000, 2000, 3000] 
options = struct("population_size",40, "select_cutoff",20, "maximize",true, "rankweight",true, "rankbasis", true,...
        "sphere_norm", 300, "lr",1.5, "mu_init", 50, "mu_final", 7.33, "indegree", true);
optimizer = ZOHA_Sphere_lr_euclid_ReducDim(4096, bn, options);
optimizer.getBasis(eigvect_avg(end-ofs-bn+1:end-ofs,:));% this is a bug.... each column is a eigen vector not each row. 
optimizer.lr_schedule(100, "exp");
param_lab = compose("nullsp%d_ofs%d", bn, ofs);
[scores_all,codes_all,generations,norm_all,Optimizer,h,exp_id,exp_dir] = GANoptim_test_fun(unit, 100, optimizer, ...
    init_genes, nsr, h, param_lab, savedir);
toc
end
% sample the same number of dimensions randomly
optimizer = ZOHA_Sphere_lr_euclid_ReducDim(4096, bn, options);
optimizer.getBasis("rand");
optimizer.lr_schedule(100, "exp");
param_lab = compose("rnd%d", bn);
[scores_all,codes_all,generations,norm_all,Optimizer,h,exp_id,exp_dir] = GANoptim_test_fun(unit, 100, optimizer, ...
    init_genes, nsr, h, param_lab, savedir);
basis = optimizer.basis;
save(fullfile(exp_dir, compose("basis_%s_%s_%s_%d.mat", class(Optimizer), param_lab, num2str(nsr), exp_id)), "basis")
toc
end

options = struct("population_size",40, "select_cutoff",20, "maximize",true, "rankweight",true, "rankbasis", true,...
        "sphere_norm", 300, "lr",1.5, "mu_init", 50, "mu_final", 7.33, "indegree", true);
optimizer = ZOHA_Sphere_lr_euclid(4096, options);
optimizer.lr_schedule(100, "exp");
param_lab = "full_ref40";
[scores_all,codes_all,generations,norm_all,Optimizer,h,exp_id,exp_dir] = GANoptim_test_fun(unit, 100, optimizer, ...
    init_genes, nsr, h, param_lab, savedir);
toc
end
end
%% Load the simulation (Old version)
unit_list = {{"fc8",2}, {"conv5",2,[7,7]}, {"conv2",2,[14,14]}};
savedir = "E:\OneDrive - Washington University in St. Louis\HessTune\NullSpace\Optim";
nsr = 0;
bn_list = [50, 100, 200, 400];
ofs_list = [0, 100, 500, 1000, 2000, 3000];
score_sum_table = nan(3,4,7,4);
for uniti = 1:numel(unit_list)
unit = unit_list{uniti};
[my_layer, iChan] = unit{1:2};
if contains(my_layer, "fc"), i=1; j=1; else, i=unit{3}(1); j=unit{3}(2);end 
exp_dir = fullfile(savedir, sprintf('%s_%d_(%d,%d)',my_layer, iChan, i, j) );

optim_str = "ZOHA_Sphere_lr_euclid_ReducDim";
for bni = 1:numel(bn_list)
bn = bn_list(bni);
for ofsi = 1:numel(ofs_list)
ofs = ofs_list(ofsi);
param_lab = compose("nullsp%d_ofs%d", bn, ofs);
fnlist = string(ls(fullfile(fullfile(exp_dir, sprintf("Evol_Data_%s%s_%s_tr*.mat", ...
                optim_str, param_lab, num2str(nsr))))));
for fni = 1:numel(fnlist)
    load(fullfile(exp_dir, fnlist(fni)), "scores_fin","codes_fin")
    score_sum_table(uniti, bni, ofsi, fni) = mean(scores_fin);
end
end

param_lab = compose("rnd%d", bn);
fnlist = string(ls(fullfile(fullfile(exp_dir, sprintf("Evol_Data_%s%s_%s_tr*.mat", ...
                optim_str, param_lab, num2str(nsr))))));
for fni = 1:numel(fnlist)
    load(fullfile(exp_dir, fnlist(fni)), "scores_fin","codes_fin")
    score_sum_table(uniti, bni, length(ofs_list)+1, fni) = mean(scores_fin);
end
end

param_lab = "full_ref40";
optim_str = "ZOHA_Sphere_lr_euclid";
fnlist = string(ls(fullfile(fullfile(exp_dir, sprintf("Evol_Data_%s%s_%s_tr*.mat", ...
                optim_str, param_lab, num2str(nsr))))));
for fni = 1:numel(fnlist)
    load(fullfile(exp_dir, fnlist(fni)), "scores_fin","codes_fin")
    ref_score(uniti, fni) = mean(scores_fin);
end
end
%% Visualize the results 
score_sum_table(score_sum_table==0)=nan;
meanscore = nanmean(score_sum_table,4);
meanscore2d = reshape(permute(meanscore, [1,3,2]), 3, []);
refmean = nanmean(ref_score,2);
figure(5);imagesc(1:28,1:3,meanscore2d./refmean)
yticks([1,2,3]);yticklabels(cellfun(@(U)U{1},unit_list))
xticks(1:28)
xticklabels(["1-50", "101-150","501-550","1001-1050","2001-2050","3001-3050","rnd50",...
             "1-100","101-200","501-600","1001-1100","2001-2100","3001-3100","rnd100",...
             "1-200","101-300","501-700","1001-1200","2001-2200","3001-3200","rnd200",...
             "1-400","101-500","501-900","1001-1400","2001-2400","3001-3400","rnd400"])
xtickangle(20)
ylabel("Target Layers in AlexNet")
xlabel("Different Subspaces")
colorbar()
title("Effect of different subspace on activation")
% savefig(3,fullfile(savedir,"subsp_cmp_pcolor.fig"))
% saveas(3,fullfile(savedir,"subsp_cmp_pcolor.jpg"))
%%
figure(4);
bar(meanscore./meanscore(:,9))
xticklabels(cellfun(@(U)U{1},unit_list))
xlabel("Target Layers in AlexNet")
ylabel("Activation (fraction of full space)")
title("Effect of different subspace on activation")
savefig(4,fullfile(savedir,"subsp_cmp_bar.fig"))
saveas(4,fullfile(savedir,"subsp_cmp_bar.jpg"))