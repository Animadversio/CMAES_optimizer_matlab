
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
addpath geomtry_util\
%%
savedir = "E:\OneDrive - Washington University in St. Louis\HessTune\NullSpace\Optim";
init_genes=randn(30,4096);nsr=0;
bn=100;
optimizer = CMAES_ReducDim(4096, [], bn, struct("init_sigma", 4, "Aupdate_freq",200));
optimizer.getBasis(eigvect_avg(end-bn+1:end,:));
unit = {"fc8",2};
[scores_all,codes_all,generations,norm_all,Optimizer,h,exp_id] = GANoptim_test_fun(unit, 100, optimizer, ...
    init_genes, nsr, [], "nullspace", savedir);
%%
unit_list = {{"fc8",2}, {"fc7",2}, {"fc6",2}, ...
    {"conv5",2,[7,7]}, {"conv4",2,[7,7]}, {"conv3",2,[7,7]}, {"conv2",2,[14,14]}, {"conv1",2,[28,28]}};
savedir = "E:\OneDrive - Washington University in St. Louis\HessTune\NullSpace\Optim";
init_genes=randn(30,4096);nsr=0;
for uniti = 1:numel(unit_list)
unit = unit_list{uniti};
tic
for triali = 1:3

for bn=[50, 100, 200, 400]
options = struct("population_size",40, "select_cutoff",15, "maximize",true, "rankweight",true, "rankbasis", true,...
        "sphere_norm", 300, "lr",1.5, "mu_init", 50, "mu_final", 7.33, "indegree", true);
optimizer = ZOHA_Sphere_lr_euclid_ReducDim(4096, bn, options);
optimizer.getBasis(eigvect_avg(end-bn+1:end,:));
optimizer.lr_schedule(100, "exp");
param_lab = compose("nullsp%d", bn);
[scores_all,codes_all,generations,norm_all,Optimizer,h,exp_id,exp_dir] = GANoptim_test_fun(unit, 100, optimizer, ...
    init_genes, nsr, [], param_lab, savedir);
toc
optimizer = ZOHA_Sphere_lr_euclid_ReducDim(4096, bn, options);
optimizer.getBasis("rand");
optimizer.lr_schedule(100, "exp");
param_lab = compose("rnd%d", bn);
[scores_all,codes_all,generations,norm_all,Optimizer,h,exp_id,exp_dir] = GANoptim_test_fun(unit, 100, optimizer, ...
    init_genes, nsr, [], param_lab, savedir);
basis = optimizer.basis;
save(fullfile(exp_dir, compose("basis_%s_%s_%s_%d.mat", class(Optimizer), param_lab, num2str(nsr), exp_id)), "basis")
toc
end

options = struct("population_size",40, "select_cutoff",15, "maximize",true, "rankweight",true, "rankbasis", true,...
        "sphere_norm", 300, "lr",1.5, "mu_init", 50, "mu_final", 7.33, "indegree", true);
optimizer = ZOHA_Sphere_lr_euclid(4096, options);
optimizer.lr_schedule(100, "exp");
param_lab = "full_ref40";
[scores_all,codes_all,generations,norm_all,Optimizer,h,exp_id,exp_dir] = GANoptim_test_fun(unit, 100, optimizer, ...
    init_genes, nsr, [], param_lab, savedir);
toc
end
end
%%
unit_list = {{"fc8",2}, {"fc7",2}, {"fc6",2}, ...
    {"conv5",2,[7,7]}, {"conv4",2,[7,7]}, {"conv3",2,[7,7]}, {"conv2",2,[14,14]}, {"conv1",2,[28,28]}};
savedir = "E:\OneDrive - Washington University in St. Louis\HessTune\NullSpace\Optim";
nsr = 0;
bn_list = [50, 100, 200, 400];
score_sum_table = nan(8,10,3);
for uniti = 1:numel(unit_list)
unit = unit_list{uniti};
[my_layer, iChan] = unit{1:2};
if contains(my_layer, "fc"), i=1; j=1; else, i=unit{3}(1); j=unit{3}(2);end 
exp_dir = fullfile(savedir, sprintf('%s_%d_(%d,%d)',my_layer, iChan, i, j) );
for bni = 1:numel(bn_list)
bn = bn_list(bni);
optim_str = "ZOHA_Sphere_lr_euclid_ReducDim";
param_lab = compose("nullsp%d", bn);
fnlist = string(ls(fullfile(fullfile(exp_dir, sprintf("Evol_Data_%s%s_%s_tr*.mat", ...
                optim_str, param_lab, num2str(nsr))))));
for fni = 1:numel(fnlist)
    load(fullfile(exp_dir, fnlist(fni)), "scores_fin","codes_fin")
    score_sum_table(uniti, bni, fni) = mean(scores_fin);
end
optim_str = "ZOHA_Sphere_lr_euclid_ReducDim";
param_lab = compose("rnd%d", bn);
fnlist = string(ls(fullfile(fullfile(exp_dir, sprintf("Evol_Data_%s%s_%s_tr*.mat", ...
                optim_str, param_lab, num2str(nsr))))));
for fni = 1:numel(fnlist)
    load(fullfile(exp_dir, fnlist(fni)), "scores_fin","codes_fin")
    score_sum_table(uniti, length(bn_list)+bni, fni) = mean(scores_fin);
end
end

param_lab = "full_ref";
optim_str = "ZOHA_Sphere_lr_euclid";
fnlist = string(ls(fullfile(fullfile(exp_dir, sprintf("Evol_Data_%s%s_%s_tr*.mat", ...
                optim_str, param_lab, num2str(nsr))))));
for fni = 1:numel(fnlist)
    load(fullfile(exp_dir, fnlist(fni)), "scores_fin","codes_fin")
    score_sum_table(uniti, 2*length(bn_list)+1, fni) = mean(scores_fin);
end

param_lab = "full_ref40";
optim_str = "ZOHA_Sphere_lr_euclid";
fnlist = string(ls(fullfile(fullfile(exp_dir, sprintf("Evol_Data_%s%s_%s_tr*.mat", ...
                optim_str, param_lab, num2str(nsr))))));
for fni = 1:numel(fnlist)
    load(fullfile(exp_dir, fnlist(fni)), "scores_fin","codes_fin")
    score_sum_table(uniti, 2*length(bn_list)+2, fni) = mean(scores_fin);
end
end
%%
score_sum_table(score_sum_table==0)=nan;
meanscore = nanmean(score_sum_table,3);
figure(3);imagesc(meanscore./meanscore(:,9))
yticklabels(cellfun(@(U)U{1},unit_list))
xticklabels(["H eig 50d", "H eig 100d", "H eig 200d", "H eig 400d", ...
             "rand 50d", "rand 100d", "rand 200d", "rand 400d", "full", "full40"])
xtickangle(20)
ylabel("Target Layers in AlexNet")
xlabel("Different Subspaces")
colorbar()
title("Effect of different subspace on activation")
savefig(3,fullfile(savedir,"subsp_cmp_pcolor.fig"))
saveas(3,fullfile(savedir,"subsp_cmp_pcolor.jpg"))
%%
figure(4);
bar(meanscore./meanscore(:,9))
xticklabels(cellfun(@(U)U{1},unit_list))
xlabel("Target Layers in AlexNet")
ylabel("Activation (fraction of full space)")
title("Effect of different subspace on activation")
savefig(4,fullfile(savedir,"subsp_cmp_bar.fig"))
saveas(4,fullfile(savedir,"subsp_cmp_bar.jpg"))