%% Testing Optimizers on Real GANs
% declare net
% net = alexnet;
grad_vect = randn(1, dimen);
score_lin = @(x) x * grad_vect';
%%
encode_vect = 5 * randn(1, dimen);
sigma = 10000;
Amp = 100;
score_guass = @(x) Amp * exp(- sum((x - encode_vect).^2, 2)/sigma);
%%
pic_size = net.Layers(1).InputSize;
% declare your generator
GANpath = "D:\Github\Monkey_Visual_Experiment_Data_Processing\DNN";
addpath(GANpath)
G = FC6Generator('matlabGANfc6.mat');
my_final_path =  '\\storage1.ris.wustl.edu\crponce\Active\Data-Computational\Project_Optimizers';
%%
% Set options for optimizer (There is default value, so it can run with empty structure)
% options = struct("population_size",40, "select_cutoff",20, "lr",2, "mu",0.005, "Lambda",1, ...
%         "Hupdate_freq",201, "maximize",true, "sphere_norm",300, "rankweight",true, "rankbasis", false, "nat_grad",false);
% Optimizer = ZOHA_Sphere(4096, options);

% options = struct("population_size",40, "select_cutoff",20, "lr_sph",2, "mu_sph",0.005, "lr_norm", 1, "mu_norm", 20, "Lambda",1, ...
%         "Hupdate_freq",201, "maximize",true, "max_norm",800, "rankweight",true, "rankbasis", false, "nat_grad",false);
% Optimizer = ZOHA_Cylind(4096, options);

% options = struct("population_size",40, "select_cutoff",20, "lr_sph",2, "mu_sph",0.005, "lr_norm", 5, "mu_norm", 5, "nu_norm", 0.95, "Lambda",1, ...
%         "Hupdate_freq",201, "maximize",true, "max_norm",800, "rankweight",true, "rankbasis", false, "nat_grad",false);
% Optimizer = ZOHA_Cylind_normmom(4096, options);

% options = struct("population_size",40, "select_cutoff",20, "lr_sph",2, "mu_sph",0.045, "lr_norm", 5, "mu_norm", 10, "Lambda",1, ...
%         "Hupdate_freq",201, "maximize",true, "max_norm",800, "rankweight",true, "rankbasis", false, "nat_grad",false);
% Optimizer = ZOHA_Cylind_ReducDim(4096, 50, options);
% Optimizer.getBasis("rand");

% Optimizer = CMAES_ReducDim(4096, [], 50);
% Optimizer.getBasis("rand");
% Optimizer =  GA_classic(randn(31, 4096), []);
Optimizer =  CMAES_simple(4096, [], struct());
n_gen = 100 ; % declare your number of generations
% unit = {"conv2", 2, 100};
Visualize = true;
SaveImg = false;   
SaveData = true; 
options = Optimizer.opts; % updates the default parameters
options.Optimizer = class(Optimizer);   
score_fun = score_guass; % score_lin
%%
% define random set of input vectors (samples from a gaussian, 0, -1)
% 30 x 4096
%init_genes = normrnd(0,1,30,4096) * 4;% have to make sure the first row cannot be all 0. 
% normrnd(0,1,30,4096) * 9.04;
genes = normrnd(0,1,30,4096) * 4;
init_genes = [mean(genes, 1) ; genes]; % have to make sure the first row cannot be all 0. 
scatclr = "cyan";%[0.8500, 0.3250, 0.0980];
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

exp_dir = fullfile(my_final_path, sprintf('Null_Gauss') );
if ~exist(exp_dir,'dir')
    mkdir(exp_dir)
end
fprintf(exp_dir)
fprintf(printOptionStr(options))
%    evolutions
genes = init_genes; 
codes_all = [];
scores_all = [];
generations = [];
mean_activation = nan(1,n_gen) ;
for iGen = 1:n_gen
    % generate pictures
    pics = G.visualize(genes);
    % feed them into net
    pics = imresize( pics , [pic_size(1) pic_size(2)]);
    % get activations
    act_unit = squeeze( score_fun(genes) ) ;
    disp(act_unit')
    % Record info 
    scores_all = [scores_all; act_unit]; 
    codes_all = [codes_all; genes];
    generations = [generations; iGen * ones(length(act_unit), 1)];
    % pass that unit's activations into CMAES_simple
    % save the new codes as 'genes'
    [genes_new,tids] = Optimizer.doScoring(genes, act_unit, true, struct());
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
    fid = fopen(fullfile(exp_dir, sprintf('parameter_%s_tr%04d.txt', class(Optimizer), exp_id)), 'wt');
    fprintf(fid, printOptionStr(options));
    fprintf("\n");
    fclose(fid);
    save(fullfile(exp_dir, sprintf("Evol_Data_%s_tr%04d.mat", class(Optimizer), exp_id)), "scores_all","codes_all","generations","norm_all")
    saveas(h, fullfile(exp_dir, sprintf("Evol_trace_%s_tr%04d.png", class(Optimizer), exp_id)))
end
%%
[~,idx]=max(scores_all);
imwrite( G.visualize(codes_all(idx, :)), sprintf('%s_%02d_%d.jpg',my_layer, iChan, t_unit) , 'jpg')
BestImg = G.visualize(codes_all(idx, :));