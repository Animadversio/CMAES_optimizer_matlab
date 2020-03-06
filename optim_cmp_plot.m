%% Plot 

result_dir = "C:\Users\ponce\OneDrive - Washington University in St. Louis\Optimizer_Tuning\noise_test\fc8_2_1";
output_dir = "C:\Users\ponce\OneDrive - Washington University in St. Louis\Optimizer_Tuning\noise_test";
%%
optim_strs = ["CMAES_simple", "ZOHA_Sphere", "ZOHA_Cylind", "ZOHA_Cylind_normmom"];
nsr_list = [0.0, 0.1, 0.2, 0.4, 0.6];
param_fn = ls(fullfile(result_dir, sprintf("parameter_%s_%s_*.txt", "ZOHA_Cylind", num2str(0.2))));
data_fn = ls(fullfile(result_dir, sprintf("Evol_Data_%s_%s_*.mat", "ZOHA_Cylind", num2str(0.2))));
D = load(fullfile(result_dir, data_fn),"generations", "scores_all", "codes_all");
%% 
% Collect the score data, multiple trials
result_dir = "C:\Users\ponce\OneDrive - Washington University in St. Louis\Optimizer_Tuning\noise_test\fc8_2_1";
optim_strs = ["CMAES_simple", "ZOHA_Sphere", "ZOHA_Cylind", "ZOHA_Cylind_normmom","GA_classic"];
nsr_list = [0.0, 0.1, 0.2, 0.4, 0.6];
score_col = {};
for optim_i = 1:length(optim_strs)
    for nsr_j = 1:length(nsr_list)
    param_fn = ls(fullfile(result_dir, sprintf("parameter_%s_%s_*.txt", optim_strs(optim_i), num2str(nsr_list(nsr_j)))));
    data_fn  = ls(fullfile(result_dir, sprintf("Evol_Data_%s_%s_*.mat", optim_strs(optim_i), num2str(nsr_list(nsr_j)))));
    data_fn = string(data_fn); param_fn = string(param_fn);
    for trial_k = 1:length(data_fn)
    D = load(fullfile(result_dir, data_fn{trial_k}),"generations", "scores_all", "norm_all");
    max_gen = max(D.generations);
    final_scores = D.scores_all(D.generations > max_gen - 5);
    score_col{optim_i, nsr_j, trial_k} = final_scores;
    end
    end
end
%%
optim_strs = ["CMAES_simple", "ZOHA_Sphere", "ZOHA_Cylind", "ZOHA_Cylind_normmom","GA_classic","ZOHA_Cylind_lr"];
result_dir = "C:\Users\ponce\OneDrive - Washington University in St. Louis\Optimizer_Tuning\lrsched_test\fc8_2_1";
for optim_i = 6
    for nsr_j = 1:length(nsr_list)
    param_fn = ls(fullfile(result_dir, sprintf("parameter_%s_%s_*.txt", optim_strs(optim_i), num2str(nsr_list(nsr_j)))));
    data_fn  = ls(fullfile(result_dir, sprintf("Evol_Data_%s_%s_*.mat", optim_strs(optim_i), num2str(nsr_list(nsr_j)))));
    data_fn = string(data_fn); param_fn = string(param_fn);
    for trial_k = 1:length(data_fn)
    D = load(fullfile(result_dir, data_fn{trial_k}),"generations", "scores_all", "norm_all");
    max_gen = max(D.generations);
    final_scores = D.scores_all(D.generations > max_gen - 5);
    score_col{optim_i, nsr_j, trial_k} = final_scores;
    end
    end
end
%% Fetch the Score Traces 
optim_strs = ["CMAES_simple", "ZOHA_Sphere", "ZOHA_Cylind", "ZOHA_Cylind_normmom","GA_classic"];
nsr_list = [0.0, 0.1, 0.2, 0.4, 0.6];
score_traj_col = cell(length(optim_strs),length(nsr_list));
for optim_i = 1:length(optim_strs)
    for nsr_j = 1:length(nsr_list)
    param_fn = ls(fullfile(result_dir, sprintf("parameter_%s_%s_*.txt", optim_strs(optim_i), num2str(nsr_list(nsr_j)))));
    data_fn  = ls(fullfile(result_dir, sprintf("Evol_Data_%s_%s_*.mat", optim_strs(optim_i), num2str(nsr_list(nsr_j)))));
    D = load(fullfile(result_dir, data_fn),"generations", "scores_all", "norm_all");
    max_gen = max(D.generations);
    score_mean = zeros(1, max_gen);
    score_std = zeros(1, max_gen);
    for geni=1:score_mean
        score_mean(geni) = mean(D.scores_all(D.generations == geni));
        score_std(geni) = std(D.scores_all(D.generations == geni));
    end
    score_traj_col{optim_i, nsr_j} = [score_mean; score_std];
    end
end
%% Plot the score trace 

%% Plot the Mean and Max score of last few generations together. 

score_col(cellfun(@isempty, score_col)) = {nan}; % turn empty cells to nan.
optim_names = cellfun(@(c) strrep(c, "_"," "), optim_strs);
meanscore = cellfun(@double, cellfun(@mean, score_col, "UniformOutput", false));
maxscore = cellfun(@double, cellfun(@max, score_col, "UniformOutput", false));
[XX,YY]=meshgrid(1:length(nsr_list),1:length(optim_strs));
h=figure(1);h.Position = [532         351        1733         627];
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
title("Mean score of last 5 generations")
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
title("Max score of last 5 generations")
axis equal tight
colorbar()
saveas(h, fullfile(output_dir, "optim_perform_noiselev.png"))
% save_to_pdf(h, fullfile(output_dir, "optim_perform_noiselev.pdf"))
