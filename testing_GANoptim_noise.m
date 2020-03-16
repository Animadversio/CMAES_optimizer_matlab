%% Testing Optimizers on Real GANs
% declare net
net = alexnet;
pic_size = net.Layers(1).InputSize;
% declare your generator
GANpath = "D:\Github\Monkey_Visual_Experiment_Data_Processing\DNN";
GANpath = "C:\Users\binxu\Documents\Monkey_Visual_Experiment_Data_Processing\DNN";
addpath(GANpath)
G = FC6Generator('matlabGANfc6.mat');
% my_final_path =  '\\storage1.ris.wustl.edu\crponce\Active\Data-Computational\Project_Optimizers';
%%
my_final_path = "C:\Users\binxu\OneDrive - Washington University in St. Louis\Optimizer_Tuning\lrsched_test";
% Set options for optimizer (There is default value, so it can run with empty structure)

% options = struct("population_size",40, "select_cutoff",20, "lr",2, "mu",0.005, "Lambda",1, ...
%         "Hupdate_freq",201, "maximize",true, "sphere_norm",350, "rankweight",true, "rankbasis", false, "nat_grad",false);
% Optimizer = ZOHA_Sphere(4096, options);
% options = struct("population_size",40, "select_cutoff",20, "lr_sph",2, "mu_sph",0.005, "lr_norm", 5, "mu_norm", 10, "Lambda",1, ...
%         "Hupdate_freq",201, "maximize",true, "max_norm",800, "rankweight",true, "rankbasis", false, "nat_grad",false);
% Optimizer = ZOHA_Cylind(4096, options);

% options = struct("population_size",40, "select_cutoff",20, "lr_sph",2, "mu_sph",0.005, "lr_norm", 5, "mu_norm", 5, "nu_norm", 0.95, "Lambda",1, ...
%         "Hupdate_freq",201, "maximize",true, "max_norm",800, "rankweight",true, "rankbasis", false, "nat_grad",false);
% Optimizer = ZOHA_Cylind_normmom(4096, options);

% options = struct("population_size",40, "select_cutoff",20, "lr_sph",2, "mu_sph",0.045, "lr_norm", 5, "mu_norm", 10, "Lambda",1, ...
%         "Hupdate_freq",201, "maximize",true, "max_norm",800, "rankweight",true, "rankbasis", false, "nat_grad",false);
% Optimizer = ZOHA_Cylind_ReducDim(4096, 50, options);
% Optimizer.getBasis("rand");
% 
% Optimizer = CMAES_ReducDim(4096, [], 50);
% Optimizer.getBasis("rand");

% Optimizer = CMAES_ReducDim(4096, [], 50);
% Optimizer.getBasis("rand");

% Optimizer =  CMAES_simple(4096, [], struct());
% Optimizers{1} = CMAES_simple(4096, [], struct());
% options = struct("population_size",40, "select_cutoff",20, "lr",2, "mu",0.005, "Lambda",1, ...
%         "Hupdate_freq",201, "maximize",true, "sphere_norm",350, "rankweight",true, "rankbasis", false, "nat_grad",false);
% Optimizers{2} = ZOHA_Sphere(4096, options);
% options = struct("population_size",40, "select_cutoff",20, "lr_sph",2, "mu_sph",0.005, "lr_norm", 5, "mu_norm", 10, "Lambda",1, ...
%         "Hupdate_freq",201, "maximize",true, "max_norm",800, "rankweight",true, "rankbasis", false, "nat_grad",false);
% Optimizers{3} = ZOHA_Cylind(4096, options);
% options = struct("population_size",40, "select_cutoff",20, "lr_sph",2, "mu_sph",0.005, "lr_norm", 5, "mu_norm", 5, "nu_norm", 0.95, "Lambda",1, ...
%         "Hupdate_freq",201, "maximize",true, "max_norm",800, "rankweight",true, "rankbasis", false, "nat_grad",false);
% Optimizers{4} = ZOHA_Cylind_normmom(4096, options);
%%
score_col = {};
mid_score_col = {};
pre_score_col = {};
n_gen = 100 ; % declare your number of generations
unit = {"fc8", 2}; % Select target unit 
Visualize = true;
SaveImg = false;   
SaveData = true; 
for trial_k = 1:5
nsr_list = [0, 0.1, 0.2, 0.4, 0.6];
for nsr_j = 1:5
nsr = nsr_list(nsr_j);
for optim_i=1%:length(Optimizers)
    %Optimizer = Optimizers{i};
options = struct("population_size",40, "select_cutoff",20, "maximize",true, "rankweight",true, "rankbasis", true,...
        "sphere_norm", 300, "lr",1.5, "mu_init", 50, "mu_final", 7.33, "indegree", true);
Optimizer = ZOHA_Sphere_lr_euclid(4096, options);
Optimizer.lr_schedule(100, "exp");
param_lab = "Lmu";
%%
% unit = {"conv2", 2, 100};
options = Optimizer.opts; % updates the default parameters
options.Optimizer = class(Optimizer);   
%%
% define random set of input vectors (samples from a gaussian, 0, -1)
% 30 x 4096
%init_genes = normrnd(0,1,30,4096) * 4;% have to make sure the first row cannot be all 0. 
% normrnd(0,1,30,4096) * 9.04;
genes = normrnd(0,1,30,4096) * 1;
init_genes = [mean(genes, 1) ; genes]; % have to make sure the first row cannot be all 0. 
scatclr = "blue";%[0.8500, 0.3250, 0.0980];
fign = [];
if ~isempty(fign)
    h = figure(fign);
    h.Position = [210         276        1201         645];
else
    h = figure();
    h.Position = [210         276        1201         645];
end
% Annotate the readable parameters on the figure. 
annotation(h,'textbox',...
    [0.485 0.467 0.154 0.50],'String',split(printOptionStr(options),','),...
    'FontSize',14,'FitBoxToText','on','EdgeColor','none')
% all_layers_wanted = {'conv4','conv5','conv3','conv2','conv1'};
% n_unitsInChan = 10 ;
% % select random units
% rng(1)
% all_units = randsample(nrows*ncols,n_unitsInChan,false) ;
% t_unit = all_units(iUnit) ; 
my_layer = unit{1} ; 
iChan = unit{2} ; 
if contains(my_layer,"fc")
	t_unit = 1 ;
else
	t_unit = unit{3};
end
if SaveData || SaveImg
exp_dir = fullfile(my_final_path, sprintf('%s_%d_%d',my_layer, iChan, t_unit) );
if ~exist(exp_dir,'dir')
    mkdir(exp_dir)
end
fprintf(exp_dir)
end
fprintf(printOptionStr(options))
% get activation size
act1 = activations(net,rand(pic_size),my_layer,'OutputAs','Channels');
[nrows,ncols,nchans] = size(act1) ;
[i,j] = ind2sub( [nrows ncols], t_unit ) ;
%    evolutions
genes = init_genes; 
codes_all = [];
scores_noise_all = [];
scores_all = [];
generations = [];
mean_activation = nan(1,n_gen) ;
for iGen = 1:n_gen
    % generate pictures
    pics = G.visualize(genes);
    % feed them into net
    pics = imresize( pics , [pic_size(1) pic_size(2)]);
    % get activations
    act1 = activations(net,pics,my_layer,'OutputAs','Channels');
    act_unit = squeeze( act1(i,j,iChan,:) ) ;
    act_unit_noise = act_unit .* (1 + nsr * randn(size(act_unit)));
    fprintf("Gen %d, score mean %.2f (%.2f), max %.2f\t", iGen, mean(act_unit), std(act_unit), max(act_unit)) %disp(act_unit')
    fprintf("with nsr %.1f, score mean %.2f (%.2f), max %.2f\n", nsr, mean(act_unit_noise), std(act_unit_noise), max(act_unit_noise))
    % Record info 
    scores_all = [scores_all; act_unit]; 
    codes_all = [codes_all; genes];
    scores_noise_all = [scores_noise_all; act_unit_noise];
    generations = [generations; iGen * ones(length(act_unit), 1)];
    % pass that unit's activations into CMAES_simple
    % save the new codes as 'genes'
    [genes_new,tids] = Optimizer.doScoring(genes, act_unit_noise, true, struct());
    if Visualize
    set(0,"CurrentFigure",h)
    % plot firing rate as it goes
    subplot(2,2,1)
    mean_activation(iGen) = mean(act_unit) ;
    scatter(iGen*ones(1,length(act_unit)),act_unit,16,...
        'MarkerFaceColor',scatclr,'MarkerEdgeColor',scatclr,...
        'MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2)
    plot(iGen, mean(act_unit) ,'r.','markersize',20)
    xlim([0, n_gen])
    ylabel("scores")
    xlabel("generations")
    hold on
    subplot(2,2,3)
    code_norms = sqrt(sum(genes.^2, 2));
    scatter(iGen*ones(1,length(code_norms)),code_norms,16,...
        'MarkerFaceColor',scatclr,'MarkerEdgeColor',scatclr,...
        'MarkerFaceAlpha',.4,'MarkerEdgeAlpha',.4)
    plot(iGen, mean(code_norms) ,'r.','markersize',20)
    xlim([0, n_gen])
    ylabel("code norm")
    xlabel("generations")
    hold on
    subplot(2,2,2)
    cla
    meanPic  = G.visualize(mean(genes));
    imagesc(meanPic);
    axis image off
    subplot(2,2,4)
    cla
    [mxscore, mxidx]= max(act_unit);
    maxPic  = G.visualize(genes(mxidx, :));
    imagesc(maxPic);
    if mxidx == 1
        title(sprintf("basis %s",num2str(mxscore)))
    else
        title(num2str(mxscore))
    end
    axis image off
    drawnow
	end
	if SaveImg
        image_name = sprintf('%s_%03d_%02d_%02d_%02d_%02d.jpg',my_layer,iChan,t_unit,nrows,ncols,iGen) ;
        imwrite( meanPic ,  ...
            fullfile(my_final_path, my_layer, sprintf('%02d',t_unit), image_name ) , 'jpg')
    end
    genes = genes_new;
end % of iGen
norm_all = sqrt(sum(codes_all.^2,2));
%
exp_id = randi(9999,1);
if SaveData % write the parametes strings to file.
    fid = fopen(fullfile(exp_dir, sprintf('parameter_%s%s_%s_tr%04d.txt', class(Optimizer), param_lab, num2str(nsr), exp_id)), 'wt');
    fprintf(fid, printOptionStr(options));
    fprintf("\n");
    fclose(fid);
    save(fullfile(exp_dir, sprintf("Evol_Data_%s%s_%s_tr%04d.mat", class(Optimizer), param_lab, num2str(nsr), exp_id)), "scores_all","codes_all","generations","norm_all")
    saveas(h, fullfile(exp_dir, sprintf("Evol_trace_%s%s_%s_tr%04d.png", class(Optimizer), param_lab, num2str(nsr), exp_id)))
end
end
score_col{optim_i, nsr_j, trial_k} = scores_all(generations>95);
mid_score_col{optim_i, nsr_j, trial_k} = scores_all(generations>25 & generations<=30);
pre_score_col{optim_i, nsr_j, trial_k} = scores_all(generations>15 & generations<=20);
end
end
save(fullfile(my_final_path, sprintf("%s%s_score_arr.mat", class(Optimizer), param_lab)),"score_col","mid_score_col","pre_score_col");
%%
[~,idx]=max(scores_all);
imwrite( G.visualize(codes_all(idx, :)), sprintf('%s_%02d_%d.jpg',my_layer, iChan, t_unit) , 'jpg')
BestImg = G.visualize(codes_all(idx, :));
