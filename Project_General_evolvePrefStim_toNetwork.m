function [meanPic , meanPic_masked,mean_activation] = Project_General_evolvePrefStim_toNetwork(t_layer,i_chan,net,my_final_path)
% Purpose: evolve best picture and activation trajectory
% Translating to work in the network.

% load('Instructions.mat');
% net_name = Instructions.net_name ;

% switch net_name
%     case 'alexnet'
% %         net = alexnet ;
%     case 'densenet'
% %         net = net2 ;
% end

cd('/scratch/crponce');

myFlags.plot = 0  ;

result = CNN_receptive_field(net) ;

% declare net
pic_size = net.Layers(1).InputSize;

% declare your number of generations
n_gen = 100 ;

% declare your generator
load('myGenerator.mat','G') % outputs G

myFlags_optimizerType = 'simple'; % 'cylinder_reduced' ; %'cylinder' ; %



% find units
test_act = activations( net, uint8( rand(pic_size) .* 255 ) , t_layer, 'OutputAs','Channels')  ;
[nrows,ncols,~] = size(test_act) ;
if prod(nrows,ncols) == 1
    % fully connected layer
    t_unit = 1 ;
else
    n_unitsInChan = nrows *ncols  ;
    % pick the unit in the middle
    t_unit = floor( n_unitsInChan / 2 ) ;
    [i,j] = ind2sub( [nrows ncols], t_unit ) ;
end

genes = normrnd(0,1,30,4096);

options = struct("population_size",40, "select_cutoff",20, "lr_sph",2, "mu_sph",0.005, "lr_norm", 5, "mu_norm", 10, "Lambda",1, ...
    "Hupdate_freq",201, "maximize",true, "max_norm",800, "rankweight",true, "rankbasis", false, "nat_grad",false);
switch myFlags_optimizerType
    case 'simple'
        C =  CMAES_simple(genes,[]);
    case 'cylinder'
        C = ZOHA_Cylind(4096, options);
    case 'cylinder_reduced'
        options = struct("population_size",40, "select_cutoff",20, "lr_sph",3, "mu_sph",0.045, "lr_norm", 5, "mu_norm", 10, "Lambda",1, ...
            "Hupdate_freq",201, "maximize",true, "max_norm",800, "rankweight",true, "rankbasis", false, "nat_grad",false);
        C = ZOHA_Cylind_ReducDim(4096, 50, options);
        C.getBasis("rand");
end


mean_activation = nan(1,n_gen) ;

% image_name = sprintf('%s_%03d_%02d_%02d_%02d_%02d.bmp',t_layer,i_chan,t_unit,nrows,ncols,1) ;

if myFlags.plot  ;  figure; end

for iGen = 1:n_gen
    
    % generate pictures
    pics = G.visualize(genes);
    % feed them into net
    pics = imresize( pics , [pic_size(1) pic_size(2)]);
    % get activations
    act1 = activations(net,pics,t_layer,'OutputAs','Channels');
    
    % pass that unit's activations into CMAES_simple
    if prod(size(act1,1:2)) > 1
        act_unit = squeeze( act1(i,j,i_chan,:) ) ;
    else
        act_unit = squeeze( act1(1,1,i_chan,:) ) ;
    end
    
    % save the new codes as 'genes'
    switch myFlags_optimizerType
        case 'simple'
            [genes,~] = C.doScoring(genes,act_unit,true);
        case 'cylinder'
            [genes,~] = C.doScoring(genes, act_unit, true, struct());
        case 'cylinder_reduced'
            [genes,~] = C.doScoring(genes, act_unit, true, struct());
    end
    mean_activation(iGen) = mean(act_unit) ;
    meanPic  = G.visualize(mean(genes));
    
    % add RF
    S_rf_i = contains( table2cell( result(:,1) ) ,  t_layer)  ;
    rf_size = table2cell( result(S_rf_i,3) ) ;
    rf_size = rf_size{1} ;
    % crop it
    midPoint = round( size(meanPic,1) / 2 ) ;  %% - rf_size(1)/2 )
    
    mask = zeros(size(meanPic),'double');
    mask = insertShape(mask,'FilledCircle',...
        [midPoint midPoint rf_size(1)],'color','w');
    mask = rescale( imgaussfilt(mask, size(mask,1)/20) ) ;
    if all(~mask(:)), mask = ones(size(mask));  end
    if max(mask(:)) > 1, mask = mask ./ max(mask(:)) ; end
    
    meanPic_masked =  mask .* im2double( meanPic ) ;
    
    image_name = sprintf('%s_%03d_%02d_%02d_%02d_%02d.bmp',t_layer,i_chan,t_unit,nrows,ncols,iGen) ;
    if ~exist(fullfile(my_final_path, t_layer ),'dir')
        mkdir(fullfile(my_final_path, t_layer ))
    end
    
    mean_activation(iGen) = mean(act_unit) ;
    meanPic  = G.visualize(mean(genes));
    
    imwrite( im2double( meanPic ) ,  ...
        fullfile(my_final_path, t_layer, image_name ) , 'bmp')
    
    if myFlags.plot
        
        % plot firing rate as it goes
        subplot(1,2,1)
        plot(iGen, mean(act_unit) ,'k.','markersize',20)
        hold on
        
        subplot(1,2,2)
        cla
        imagesc(meanPic);
        axis image off
        drawnow
        
    end % of myFlags.plot
    
end % of iGen
% savefast(
% mask .* im2double( meanPic )
image_name = sprintf('%s_%03d_%02d_%02d_%02d_%02d_full.bmp',t_layer,i_chan,t_unit,nrows,ncols,iGen) ;
imwrite( meanPic_masked ,  ...
    fullfile(my_final_path, t_layer, image_name ) , 'bmp')
image_name = sprintf('%s_%03d_%02d_%02d_%02d_%02d_mask.mat',t_layer,i_chan,t_unit,nrows,ncols,iGen) ;
im = im2double( meanPic );
savefast(fullfile(my_final_path, t_layer, image_name ),...
    'mask','im','mean_activation')
fprintf('\n')