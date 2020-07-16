function [scores_all,codes_all,generations,norm_all,Optimizer,h,exp_id,exp_dir] = GANoptim_test_fun(unit, ...
    n_gen, Optimizer, init_genes, nsr, fign, param_lab, my_final_path)
% Wrapped up function for testing Optimizer performance in matlab.  
% Require global variable net (target CNN) and G the generator equiped with
% G.visualize(.) function
% Parameters: 
%    Unit: a cell array of target unit. E.g. {"fc8", 10} or {"conv4", 11, 101}
%    n_gen: how many steps / generations to go
%    Optimizer: 
%    nsr:  noise signal ratio. 0.0 is no noise 1.0 then the noise std is as
%       large as signal amplitude
%    fign: figure number to plot into or []. 
%    my_final_path: path to write data and figure to e.g. exp per layer
%       will be written in it. 
%       my_final_path = "C:\Users\binxu\OneDrive - Washington University in St. Louis\Optimizer_Tuning\lrsched_test";
Visualize = false;
Realtime = false; % true, plot the mean image max image scores and norms at each generation; false, only plot final generation
Holdon = false;
SaveImg = false; % save the mean image for each generation
SaveData = false; % save the code and scores for the last generation
scatclr = "blue"; % color of scatter plot
global G net
options = Optimizer.opts; % updates the default parameters
options.Optimizer = class(Optimizer);
if Visualize
if ~isempty(fign)
    h = figure(fign); if ~Holdon, clf(fign); end
    h.Position = [210         276        1201         645];
else
    h = figure();
    h.Position = [210         276        1201         645];
end
% Annotate the readable parameters on the figure. 
annotation(h,'textbox',...
    [0.485 0.467 0.154 0.50],'String',split(printOptionStr(options),','),...
    'FontSize',14,'FitBoxToText','on','EdgeColor','none')
else
h = [];
end
my_layer = unit{1}; 
iChan = unit{2}; 
if contains(my_layer,"fc")
	t_unit = 1;
else
	t_unit = unit{3};
end
% get activation size
pic_size = net.Layers(1).InputSize;
if length(t_unit) == 1
act1 = activations(net,rand(pic_size),my_layer,'OutputAs','Channels');
[nrows,ncols,nchans] = size(act1) ;
[i,j] = ind2sub( [nrows ncols], t_unit ) ;
else
i = t_unit(1); j = t_unit(2);
end
if SaveData || SaveImg || Visualize
exp_dir = fullfile(my_final_path, sprintf('%s_%d_(%d,%d)',my_layer, iChan, i, j) );
if ~exist(exp_dir,'dir')
    mkdir(exp_dir)
end
fprintf(exp_dir)
else
exp_dir = "";
end
fprintf(printOptionStr(options))

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
    if Visualize && Realtime% plot firing rate as it goes
        set(0,"CurrentFigure",h)
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
        meanPic  = G.visualize(mean(genes,1));
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

if Visualize && (~Realtime) % post hoc plotting if Realtime is switched off.
set(0,"CurrentFigure",h)
subplot(2,2,1);hold on 
scatter(generations,scores_all,16,...
    'MarkerFaceColor',scatclr,'MarkerEdgeColor',scatclr,...
    'MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2)
plot(1:n_gen, mean_activation ,'r.','markersize',20)
xlim([0, n_gen]);ylabel("scores");xlabel("generations")

subplot(2,2,3);
% % yyaxis left
scatter(generations,norm_all,16,...
    'MarkerFaceColor',scatclr,'MarkerEdgeColor',scatclr,...
    'MarkerFaceAlpha',.4,'MarkerEdgeAlpha',.4);hold on 
plot(1:n_gen, arrayfun(@(igen) mean(norm_all(generations==igen)), 1:n_gen),'r.','markersize',20)
xlim([0, n_gen]);ylabel("code norm");xlabel("generations")
% % yyaxis right
% plot(1:n_gen, Optimizer.mulist(1:n_gen) / pi * 180 * sqrt(Optimizer.dimen))
% ylabel("angular mu")
subplot(2,2,2)
meanPic  = G.visualize(mean(genes));
imagesc(meanPic);
axis image off
subplot(2,2,4)
[mxscore, mxidx]= max(act_unit);
maxPic  = G.visualize(genes(mxidx, :));
imagesc(maxPic);
axis image off
if mxidx == 1
    title(sprintf("basis %s",num2str(mxscore)))
else
    title(num2str(mxscore))
end
% drawnow
pause(0.5)
end

exp_id = randi(9999,1);
if SaveData % write the parametes strings to file.
    fid = fopen(fullfile(exp_dir, sprintf('parameter_%s_%s_tr%04d.txt', class(Optimizer), num2str(nsr), exp_id)), 'wt');
    fprintf(fid, printOptionStr(options));
    fprintf("\n");
    fclose(fid);
    scores_fin = scores_all(generations==n_gen,:);
    codes_fin = codes_all(generations==n_gen,:);
    save(fullfile(exp_dir, sprintf("Evol_Data_%s%s_%s_tr%04d.mat", ...
        class(Optimizer), param_lab, num2str(nsr), exp_id)), "scores_fin","codes_fin") % only save the last part of data. 
%     save(fullfile(exp_dir, sprintf("Evol_Data_%s%s_%s_tr%04d.mat", class(Optimizer), param_lab, num2str(nsr), exp_id)), "scores_all","codes_all","generations","norm_all")
end
if Visualize
saveas(h, fullfile(exp_dir, sprintf("Evol_trace_%s%s_%s_tr%04d.png", class(Optimizer), param_lab, num2str(nsr), exp_id)))
end
end