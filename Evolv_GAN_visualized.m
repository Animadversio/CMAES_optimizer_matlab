%% Testing Optimizers

% declare net
net = alexnet;
pic_size = net.Layers(1).InputSize;
% declare your number of generations
n_gen = 100 ;
% declare your generator
GANpath = ""
addpath(GANpath)
G = FC6Generator('matlabGANfc6.mat') ;
my_final_path =  '\\storage1.ris.wustl.edu\crponce\Active\Data-Computational\Project_RedDim';
%%    evolutions
% define random set of input vectors (samples from a gaussian, 0, -1)
% 30 x 4096
all_layers_wanted = {'conv4','conv5','conv3','conv2','conv1'};
all_layers_present = cell(length(net.Layers),1);
for iLayers = 1:length(net.Layers)
    all_layers_present{iLayers,1} = net.Layers(iLayers).Name ;
end

n_unitsInChan = 10 ;

for iLayer = 1%:length(all_layers_wanted)
    % declare layer
    my_layer = all_layers_wanted{iLayer} ;
    S_layer_i = strcmp(all_layers_present,my_layer) ;
    
    % get activation size
    act1 = activations(net,rand(pic_size),my_layer,'OutputAs','Channels');
    [nrows,ncols,nchans] = size(act1) ;
    
    % select random units
    rng(1)
    all_units = randsample(nrows*ncols,n_unitsInChan,false) ;
    
    for iChan = 112%1:100%:nchans
        % declare your unit
        for iUnit = 1:n_unitsInChan
            t_unit = all_units(iUnit) ;
            [i,j] = ind2sub( [nrows ncols], t_unit ) ;
            genes = normrnd(0,1,30,4096);
            Optimizer =  CMAES_simple(genes,[]);
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
                
                % pass that unit's activations into CMAES_simple
                act_unit = squeeze( act1(i,j,iChan,:) ) ;
                % save the new codes as 'genes'
                [genes,tids] = Optimizer.doScoring(genes, act_unit, true);
                % plot firing rate as it goes
                subplot(1,2,1)
                mean_activation(iGen) = mean(act_unit) ;
                plot(iGen, mean(act_unit) ,'k.','markersize',20)
                hold on
                subplot(1,2,2)
                cla
                meanPic  = G.visualize(mean(genes));
                imagesc(meanPic);
                axis image off
                drawnow
                
                image_name = sprintf('%s_%03d_%02d_%02d_%02d_%02d.bmp',my_layer,iChan,t_unit,nrows,ncols,iGen) ;
                if ~exist(fullfile(my_final_path, my_layer, sprintf('%02d',t_unit) ),'dir')
                    mkdir(fullfile(my_final_path, my_layer, sprintf('%02d',t_unit) ))
                end
                imwrite( meanPic ,  ...
                    fullfile(my_final_path, my_layer, sprintf('%02d',t_unit), image_name ) , 'bmp')
                
            end % of iGen
            save(fullfile(my_final_path, my_layer, sprintf('%02d',t_unit), image_name(1:end-4)) ,'mean_activation')
            
        end % of iUnit
    end % of iChan
end % of iLayer