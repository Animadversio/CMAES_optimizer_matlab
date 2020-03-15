global net G 
net = alexnet;
pic_size = net.Layers(1).InputSize;
% declare your generator
GANpath = "C:\Users\ponce\Documents\GitHub\Monkey_Visual_Experiment_Data_Processing\DNN";
% GANpath = "D:\Github\Monkey_Visual_Experiment_Data_Processing\DNN";
addpath(GANpath)
G = FC6Generator('matlabGANfc6.mat');
%%
score_col_1 = {}; mid_score_col_1 = {}; pre_score_col_1 = {};
n_gen = 100;
unit = {"fc8", 2};
nsr_list = [0, 0.1, 0.2, 0.4, 0.6];
for trial_k = 1%:5
for nsr_j = 1:length(nsr_list)
nsr = nsr_list(nsr_j);
options = struct("population_size",40, "select_cutoff",20, "lr",1, "Lambda",1, "sphere_norm",250, ...
        "Hupdate_freq",201, "maximize",true, "rankweight",true, "rankbasis", false, "nat_grad",false,...
         "indegree", true, "mu_init", 50, "mu_final", 7.33);
Optimizer = ZOHA_Sphere_lr(4096, options);
Optimizer.lr_schedule(n_gen,"inv");
Optimizers{1} = Optimizer;
param_lab = "invSL";
for optim_i=1:length(Optimizers)
init_genes = normrnd(0,1,30,4096);
[scores_all,codes_all,generations,norm_all,Optimizer,~] = GANoptim_test_fun(unit, n_gen, Optimizers{optim_i}, init_genes, nsr, [], param_lab);
final_scores = scores_all(generations > n_gen - 5);
mid_scores = scores_all(generations > 25 & generations <=30);
pre_scores = scores_all(generations > 15 & generations <=20);
score_col_1{optim_i, nsr_j, trial_k} = final_scores;
mid_score_col_1{optim_i, nsr_j, trial_k} = mid_scores;
pre_score_col_1{optim_i, nsr_j, trial_k} = pre_scores;
end
end
end
save(sprintf("%s_%s.mat",class(Optimizer),param_lab), "score_col_1", "mid_score_col_1", "pre_score_col_1")
%%
%% Plot the Mean and Max score of last few generations together. 
optim_strs = {"ZOHA_Sphere_lrMexp"};
h=figure(31);clf;h.Position = [ 142         542        1733         274];
plot_score_heatmap(score_col_1, nsr_list, optim_strs, "final", h)
saveas(h, fullfile(output_dir, sprintf("optim_perform_noiselev_%s.png",optim_strs{1})))
% save_to_pdf(h, fullfile(output_dir, sprintf("optim_perform_noiselev_%s.pdf",optim_strs{1})))
h=figure(32);clf;h.Position = [ 142         542        1733         274];
plot_score_heatmap(mid_score_col_1, nsr_list, optim_strs, "25-30", h)
saveas(h, fullfile(output_dir, sprintf("optim_perform_mid_noiselev_%s.png",optim_strs{1})))
h=figure(33);clf;h.Position = [ 142         542        1733         274];
plot_score_heatmap(pre_score_col_1, nsr_list, optim_strs, "15-20", h)
saveas(h, fullfile(output_dir, sprintf("optim_perform_pre_noiselev_%s.png",optim_strs{1})))

%%
function plot_score_heatmap(score_col, nsr_list, optim_strs, titlestr, h)
score_col(cellfun(@isempty, score_col)) = {nan}; % turn empty cells to nan.
optim_names = cellfun(@(c) strrep(c, "_"," "), optim_strs);
meanscore = cellfun(@double, cellfun(@mean, score_col, "UniformOutput", false));
maxscore = cellfun(@double, cellfun(@max, score_col, "UniformOutput", false));
[XX,YY]=meshgrid(1:length(nsr_list),1:length(optim_strs));
set(0,"CurrentFigure",h)
subplot(121)
imagesc(nanmean(meanscore,3))
% score_text = cellfun(@(s) num2str(s,"%.1f"), ...
%             num2cell(meanscore), 'UniformOutput', false);
score_text = cellfun(@(m,s) sprintf("%.1f\n(%.2f)", m, s), ...
            num2cell(nanmean(meanscore,3)), num2cell(nanstd(meanscore,1,3)), 'UniformOutput', false);
text(XX(:), YY(:), score_text, 'HorizontalAlignment', 'Center', "FontSize", 15)
xticks(1:length(nsr_list));  xticklabels(compose("%.1f", nsr_list))
yticks(1:length(optim_strs));yticklabels(optim_names)
ylabel("Optimizer");xlabel("noise level")
title(sprintf("Mean score of %s generations", titlestr))
axis equal tight
colorbar()
subplot(122)
% score_text = cellfun(@(s) num2str(s,"%.1f"), ...
%             num2cell(maxscore), 'UniformOutput', false);
score_text = cellfun(@(m,s) sprintf("%.1f\n(%.2f)", m, s), ...
            num2cell(nanmean(maxscore,3)), num2cell(nanstd(maxscore,1,3)), 'UniformOutput', false);
imagesc(nanmean(maxscore,3))
text(XX(:), YY(:), score_text, 'HorizontalAlignment', 'Center', "FontSize", 15)
xticks(1:length(nsr_list));  xticklabels(compose("%.1f", nsr_list))
yticks(1:length(optim_strs));yticklabels(optim_names)
ylabel("Optimizer");xlabel("noise level")
title(sprintf("Max score of %s generations", titlestr))
axis equal tight
colorbar()
end