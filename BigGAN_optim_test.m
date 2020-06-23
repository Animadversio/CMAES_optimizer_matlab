addpath E:\Github_Projects\Monkey_Visual_Experiment_Data_Processing\DNN
addpath E:\Github_Projects\Monkey_Visual_Experiment_Data_Processing\utils
addpath geomtry_util\
addpath D:\Github\Monkey_Visual_Experiment_Data_Processing\DNN
addpath D:\Github\Monkey_Visual_Experiment_Data_Processing\utils
BGAN = torchBigGAN("biggan-deep-256");
%% across class evolution
onehot = zeros(1,1000);onehot(460) = 1;
class_center = BGAN.Embeddings(py.torch.tensor(py.numpy.array(onehot)).view(int32(-1),int32(1000)).float().cuda());
cmb_z = py.torch.cat({py.torch.randn(int32(1),int32(128)).cuda(), class_center},int32(1));
img = BGAN.Generator(cmb_z,0.7);
matimg = img.detach.cpu().numpy().single;
matimg = permute((matimg + 1) / 2.0,[3,4,2,1]);
figure;imshow(matimg)
%%
% onehot = zeros(1,1000);onehot(460) = 1;
% class_center = BGAN.Embeddings(py.torch.tensor(py.numpy.array(onehot)).view(int32(-1),int32(1000)).float().cuda());
% cmb_z = py.torch.cat({py.torch.randn(int32(1),int32(128)).cuda(), class_center},int32(1));
imgs = BGAN.Generator(0.2*py.torch.randn(int32(20),int32(256)).cuda(),0.7);
matimg = imgs.detach.cpu().numpy().single;
matimg = permute((matimg + 1) / 2.0,[3,4,2,1]);
figure;montage(matimg)
%%

%%
net = alexnet;
%% Optimization in the class space. 
n_gen = 50;nsr = 0.0;Visualize = true;
unit = {"fc8",2};
% Optimization in the class space. 
% visualize_fun = @(genes) BGAN.visualize_latent([zeros(size(genes)),genes]);
% Optimizer = CMAES_simple(128, [], struct('init_sigma', 0.03)); %, struct("popsize", 40));
% init_genes =  [EmbedVects_mat(:,460)' + 0.005*randn(41,128)];

% options = struct("population_size",40, "select_cutoff",20, "maximize",true, "rankweight",true, "rankbasis", true,...
%         "sphere_norm", 0.75, "lr",1.5, "mu_init", 40, "mu_final", 15, "indegree", true);
% Optimizer = ZOHA_Sphere_lr_euclid(128, options);
% Optimizer.lr_schedule(50, "inv");
% init_genes =  [EmbedVects_mat(:,460)' + 0.005*randn(41,128)];

% Optimize in Noise space
class_id = 2;%374;
visualize_fun = @(noise) BGAN.visualize_latent([noise,EmbedVects_mat(:,class_id)'+zeros(size(noise))]);
% truncnorm = truncate(makedist("Normal"),-2,2);
init_genes =  [0.7*truncnorm.random(30,128)]; init_genes = [mean(init_genes, 1); init_genes];
options = struct("population_size",30, "select_cutoff",15, "maximize",true, "rankweight",true, "rankbasis", true,...
        "sphere_norm", 7, "lr",1.5, "mu_init", 60, "mu_final", 10, "indegree", true);
Optimizer = ZOHA_Sphere_lr_euclid(128, options);
Optimizer.lr_schedule(n_gen, "inv");
%%
options = Optimizer.opts; % updates the default parameters
options.Optimizer = class(Optimizer);
scatclr = 'blue';
h = figure(7);clf;h.Position = [210         276        1201         645];
annotation(h,'textbox',...
    [0.485 0.467 0.154 0.50],'String',split(printOptionStr(options),','),...
    'FontSize',14,'FitBoxToText','on','EdgeColor','none')
my_layer = unit{1} ; 
iChan = unit{2} ; 
if contains(my_layer,"fc")
	t_unit = 1 ;
else
	t_unit = unit{3};
end
fprintf(printOptionStr(options))
% get activation size
pic_size = net.Layers(1).InputSize;
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
%
for iGen = 1:n_gen
    % generate pictures
    pics = visualize_fun(genes);
    % feed them into net
    pics = imresize( pics , [pic_size(1) pic_size(2)]);
    % get activations
    act1 = activations(net,255*pics,my_layer,'OutputAs','Channels','ExecutionEnvironment','cpu');
    act_unit = squeeze( act1(i,j,iChan,:) ) ;
    act_unit_noise = act_unit .* (1 + nsr * randn(size(act_unit)));
    %disp(act_unit');
    %disp(act_unit_noise')
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
    meanPic = visualize_fun(mean(genes,1));
    imagesc(meanPic);
    axis image off
    subplot(2,2,4)
    cla
    [mxscore, mxidx]= max(act_unit);
    maxPic = visualize_fun(genes(mxidx, :));
    imagesc(maxPic);
    if mxidx == 1
        title(sprintf("basis %s",num2str(mxscore)))
    else
        title(num2str(mxscore))
    end
    axis image off
    drawnow
	end
% 	if SaveImg
%         image_name = sprintf('%s_%03d_%02d_%02d_%02d_%02d.jpg',my_layer,iChan,t_unit,nrows,ncols,iGen) ;
%         imwrite( meanPic ,  ...
%             fullfile(my_final_path, my_layer, sprintf('%02d',t_unit), image_name ) , 'jpg')
%     end
    genes = genes_new;
end % of iGen
norm_all = sqrt(sum(codes_all.^2,2));
%% Dual space optimization
n_gen = 20;nsr = 0.0;Visualize = true;
unit = {"fc8",2};
% Optimization in the class space. 
% visualize_fun = @(genes) BGAN.visualize_latent([zeros(size(genes)),genes]);
% Optimizer = CMAES_simple(128, [], struct('init_sigma', 0.03)); %, struct("popsize", 40));
% init_genes =  [EmbedVects_mat(:,460)' + 0.005*randn(41,128)];

% options = struct("population_size",40, "select_cutoff",20, "maximize",true, "rankweight",true, "rankbasis", true,...
%         "sphere_norm", 0.75, "lr",1.5, "mu_init", 40, "mu_final", 15, "indegree", true);
% Optimizer = ZOHA_Sphere_lr_euclid(128, options);
% Optimizer.lr_schedule(50, "inv");
% init_genes =  [EmbedVects_mat(:,460)' + 0.005*randn(41,128)];

% Optimize in Noise and class space
visualize_fun2 = @(noise,embed) BGAN.visualize_latent([noise,embed]);
truncnorm = truncate(makedist("Normal"),-2,2);
init_genes =  {[0.7*truncnorm.random(31,128)], ...
               [EmbedVects_mat(:,374)' + 0.005*randn(31,128)]};
Optimizers =  {};
options = struct("population_size",30, "select_cutoff",15, "maximize",true, "rankweight",true, "rankbasis", true,...
        "sphere_norm", 7, "lr",1.5, "mu_init", 60, "mu_final", 10, "indegree", true);
Optimizers{1} = ZOHA_Sphere_lr_euclid(128, options);
Optimizers{1}.lr_schedule(n_gen, "inv");
Optimizers{2} = CMAES_simple(128, [], struct('init_sigma', 0.06, "popsize", 31)); %, struct("popsize", 40));
% init_genes =  [EmbedVects_mat(:,460)' + 0.005*randn(41,128)];

% options = struct("population_size",30, "select_cutoff",15, "maximize",true, "rankweight",true, "rankbasis", true,...
%         "sphere_norm", 0.8, "lr",1.5, "mu_init", 40, "mu_final", 10, "indegree", true);
% Optimizers{2} = ZOHA_Sphere_lr_euclid(128, options);
% Optimizers{2}.lr_schedule(n_gen, "inv");
%%
opt_str = "";
for Oi = 1:numel(Optimizers)
options = Optimizers{Oi}.opts; % updates the default parameters
options.Optimizers = class(Optimizers{Oi});
opt_str = opt_str + printOptionStr(options) + "\n";
end
fprintf(opt_str)
scatclr = 'blue';
h = figure(9);clf;h.Position = [210         276        1201         645];
annotation(h,'textbox',...
    [0.485 0.467 0.154 0.50],'String',split(opt_str,','),...
    'FontSize',14,'FitBoxToText','on','EdgeColor','none')
my_layer = unit{1} ; 
iChan = unit{2} ; 
if contains(my_layer,"fc")
	t_unit = 1 ;
else
	t_unit = unit{3};
end
% get activation size
pic_size = net.Layers(1).InputSize;
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
%
for iGen = 1:n_gen
    % generate pictures
    pics = visualize_fun2(genes{:});
    % feed them into net
    pics = imresize( pics , [pic_size(1) pic_size(2)]);
    % get activations
    act1 = activations(net,255*pics,my_layer,'OutputAs','Channels','ExecutionEnvironment','cpu');
    act_unit = squeeze( act1(i,j,iChan,:) ) ;
    act_unit_noise = act_unit .* (1 + nsr * randn(size(act_unit)));
    %disp(act_unit');
    %disp(act_unit_noise')
    % Record info 
    scores_all = [scores_all; act_unit]; 
    codes_all = [codes_all; genes];
    scores_noise_all = [scores_noise_all; act_unit_noise];
    generations = [generations; iGen * ones(length(act_unit), 1)];
    % pass that unit's activations into CMAES_simple
    % save the new codes as 'genes'
    genes_new = {};
    [genes_new{1},tids] = Optimizers{1}.doScoring(genes{1}, act_unit_noise, true, struct());
    [genes_new{2},tids] = Optimizers{2}.doScoring(genes{2}, act_unit_noise, true, struct());
%     genes_new = {genes_new1, genes_new2};
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
    code_norms = sqrt(sum(genes{1}.^2, 2));
%     scatter(iGen*ones(1,length(code_norms)),code_norms,16,...
%         'MarkerFaceColor',scatclr,'MarkerEdgeColor',scatclr,...
%         'MarkerFaceAlpha',.4,'MarkerEdgeAlpha',.4)
    yyaxis left
    plot(iGen, mean(code_norms) ,'k.','markersize',20)
    xlim([0, n_gen])
    ylabel("code norm (noise)")
    xlabel("generations")
    yyaxis right
    code_norms = sqrt(sum(genes{2}.^2, 2));
    plot(iGen, mean(code_norms) ,'r.','markersize',20)
    ylabel("code norm (embed)")
    hold on
    subplot(2,2,2)
    cla
    meanPic = visualize_fun2(mean(genes{1},1),mean(genes{2},1));
    imagesc(meanPic);
    axis image off
    subplot(2,2,4)
    cla
    [mxscore, mxidx]= max(act_unit);
    maxPic = visualize_fun2(genes{1}(mxidx, :),genes{2}(mxidx, :));
    imagesc(maxPic);
    if mxidx == 1
        title(sprintf("basis %s",num2str(mxscore)))
    else
        title(num2str(mxscore))
    end
    axis image off
    drawnow
	end
% 	if SaveImg
%         image_name = sprintf('%s_%03d_%02d_%02d_%02d_%02d.jpg',my_layer,iChan,t_unit,nrows,ncols,iGen) ;
%         imwrite( meanPic ,  ...
%             fullfile(my_final_path, my_layer, sprintf('%02d',t_unit), image_name ) , 'jpg')
%     end
    genes = genes_new;
end % of iGen
norm_all = {};
norm_all{1} = sqrt(sum(cell2mat(codes_all(:,1)).^2,2));
norm_all{2} = sqrt(sum(cell2mat(codes_all(:,2)).^2,2));