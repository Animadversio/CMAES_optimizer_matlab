%% Testing Optimizers on Real GANs

% declare net
net = alexnet;
pic_size = net.Layers(1).InputSize;
% declare your generator
GANpath = "D:\Github\Monkey_Visual_Experiment_Data_Processing\DNN";
addpath(GANpath)
G = FC6Generator('matlabGANfc6.mat') ;
my_final_path =  '\\storage1.ris.wustl.edu\crponce\Active\Data-Computational\Project_Optimizers';
%%
% declare your number of generations
n_gen = 100 ;
unit = {"fc8", 2};
% unit = {"conv2", 10, 100};

% options = struct("population_size",40, "select_cutoff",10, "lr",2.5, "mu",0.005, "Lambda",1, ...
%         "Hupdate_freq",201, "maximize",true, "max_norm",300, "rankweight",true, "rankbasis", true, "nat_grad",false);
% Optimizer = ZOHA_Sphere(4096, options);
options = struct("population_size",40, "select_cutoff",10, "lr_sph",2.5, "mu_sph",0.005, "lr_norm", 20, "mu_norm", 5, "Lambda",1, ...
        "Hupdate_freq",201, "maximize",true, "max_norm",inf, "rankweight",true, "rankbasis", true, "nat_grad",false);
Optimizer = ZOHA_Cylind(4096, options);

% genes = normrnd(0,1,30,4096) * 9.04;
% genes = [mean(genes, 1) ; genes]; % have to make sure the first row cannot be all 0. 
% Optimizer = CMAES_ReducDim(genes, [], 50);
% Optimizer.getBasis("rand");

% Optimizer =  CMAES_simple(genes, []);
Visualize = true;
Save = false;   
fign = [];
%%
init_genes = normrnd(0,1,30,4096);
init_genes = [mean(init_genes, 1) ; init_genes]; % have to make sure the first row cannot be all 0. 
if ~isempty(fign)
    h = figure(fign);
    h.Position = [210         276        1201         645];
else
    h = figure();
    h.Position = [210         276        1201         645];
end
% subplot(2,2,1)
% subplot(2,2,3)
% xlim([0, n_gen])

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
if ~exist(fullfile(my_final_path, my_layer, sprintf('%02d',t_unit) ),'dir')
    mkdir(fullfile(my_final_path, my_layer, sprintf('%02d',t_unit) ))
end
% get activation size
act1 = activations(net,rand(pic_size),my_layer,'OutputAs','Channels');
[nrows,ncols,nchans] = size(act1) ;
[i,j] = ind2sub( [nrows ncols], t_unit ) ;
%    evolutions
% define random set of input vectors (samples from a gaussian, 0, -1)
% 30 x 4096
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
    act1 = activations(net,pics,my_layer,'OutputAs','Channels');
    act_unit = squeeze( act1(i,j,iChan,:) ) ;
    % Record info 
    scores_all = [scores_all; act_unit]; 
    codes_all = [codes_all; genes];
    generations = [generations; iGen * ones(length(act_unit), 1)];
    % pass that unit's activations into CMAES_simple
    % save the new codes as 'genes'
    [genes_new,tids] = Optimizer.doScoring(genes, act_unit, true);
    if Visualize
    % plot firing rate as it goes
    subplot(2,2,1)
    mean_activation(iGen) = mean(act_unit) ;
    scatter(iGen*ones(1,length(act_unit)),act_unit,16,...
        'MarkerFaceColor','blue','MarkerEdgeColor','blue',...
        'MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2)
    plot(iGen, mean(act_unit) ,'r.','markersize',20)
    xlim([0, n_gen])
    ylabel("scores")
    xlabel("generations")
    hold on
    subplot(2,2,3)
    code_norms = sqrt(sum(genes.^2, 2));
    scatter(iGen*ones(1,length(code_norms)),code_norms,16,...
        'MarkerFaceColor','blue','MarkerEdgeColor','blue',...
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
	if Save
    image_name = sprintf('%s_%03d_%02d_%02d_%02d_%02d.jpg',my_layer,iChan,t_unit,nrows,ncols,iGen) ;
    imwrite( meanPic ,  ...
        fullfile(my_final_path, my_layer, sprintf('%02d',t_unit), image_name ) , 'jpg')
    end
    genes = genes_new;
end % of iGen
norm_all = sqrt(sum(codes_all.^2,2));
%%
save(fullfile(my_final_path, my_layer, sprintf('%02d',t_unit)), "scores_all","codes_all","generations")
%save(fullfile(my_final_path, my_layer, sprintf('%02d',t_unit), image_name(1:end-4)) ,'mean_activation')

