myFlags.runEvolutions =         1 ;
myFlags.collectEvolutions =     0 ;
myFlags.keepChecking =          0 ;

% my_final_path =  '\\storage1.ris.wustl.edu\crponce\Active\Data-Computational\Project_Receptive';
% my_final_path =  'C:\Users\carlo\OneDrive\Desktop\tmp' ;
my_final_path =  '/scratch/crponce';
my_local_path = 'N:\Data-Computational\Project_AlexNet';

if myFlags.runEvolutions
    globalTic = tic ;
    %     clc
    % declare net
    if ~exist('net','var'); net = alexnet; end
    t_net = net ;
    net_name = 'alexnet' ;
    layers_all = {'conv1','conv2','conv3','conv4','conv5','fc6','fc7','fc8'} ;
    pic_size = net.Layers(1).InputSize;
    
    % get a list of channels available per layer, check what channels have been evolved.
    list_of_chans = nan(1,length(layers_all)) ; 
    list_of_todos = cell(1,length(layers_all)) ; 
    for iLayer = 1:length(layers_all)
        % declare layer
        my_layer = layers_all{iLayer} ;
        % get activation size
        act1 = activations(net,rand(pic_size),my_layer,'OutputAs','Channels');
        [nrows,ncols,nchans_tmp] = size(act1) ;
        list_of_chans(iLayer) = nchans_tmp ; 
        all_mat_dir = dir( fullfile(my_local_path,my_layer,'*.mat') ) ;
        all_mat_cell = {all_mat_dir.name}' ; 
        all_chans_ids = [] ; 
        for iChan = 1:length(all_mat_cell)
            t_mat_cell = all_mat_cell{iChan} ; 
            isUnder = regexp(t_mat_cell,'_')  ;
            all_chans_ids = cat(1,all_chans_ids,...
                str2double(t_mat_cell(isUnder(1)+1:isUnder(2)-1)  ) ) ;
        end
        list_of_todos{iLayer} = setdiff(1:list_of_chans(iLayer),all_chans_ids) ; 
    end
    

    % declare your generator
    G = FC6Generator('matlabGANfc6.mat') ;
    
    % get headless workers
    c = parcluster ;
    
    % channels in layer
    nchansToDo = 50 ;
    
    for iChan = 1:nchansToDo % outer loop
        
        for iLayer = 1:length(layers_all)
            
            if isempty(list_of_todos{iLayer}) , continue, end % all channels done
                
            % sample a unit from ToDo
            t_chan = datasample(list_of_todos{iLayer},1) ; 
            
            % declare layer
            my_layer = layers_all{iLayer} ;
            
            % get activation size
            act1 = activations(net,rand(pic_size),my_layer,'OutputAs','Channels');
            [nrows,ncols,nchans_tmp] = size(act1) ;
            
            % sample randomly from selection
            if prod(nrows,ncols) == 1
                t_unit = 1 ;
            else
                % pick the unit in the middle
                n_unitsInChan = nrows *ncols  ;
                % pick the unit in the middle
                t_unit = floor( n_unitsInChan / 2 ) ;
                [i,j] = ind2sub( [nrows ncols], t_unit ) ;
            end
            
            
            
            % check if it has been done:
            image_name = sprintf('%s_%03d_*.mat',my_layer,t_chan) ;
            if ~isempty( dir( fullfile(my_local_path,my_layer,image_name) ) )
                fprintf('skipping %s, already done\n',image_name)
                continue
            else
                fprintf('submitting %s\n',image_name)
            end
            
            % get activation size
            act1 = activations(net,rand(pic_size),my_layer,'OutputAs','Channels');
            [nrows,ncols,~] = size(act1) ;
            c.batch(@Project_General_evolvePrefStim_toNetwork, 3, {my_layer,t_chan,net,my_final_path}, ...
                'CurrentFolder',my_final_path) ;
            pause(2)

            %   [meanPic , meanPic_masked,mean_activation] = Project_General_evolvePrefStim_toNetwork(my_layer,t_chan,net,G,my_final_path);
            
        end % of iLayer
        
    end % of iChan
    
    if myFlags.keepChecking
        jobs = c.Jobs;
        keepChecking = 1 ;
        while keepChecking
            all_states = [];
            for i = 1:length(jobs)
                t_state = jobs(i).State ;
                all_states{i} = t_state ;
            end
            all_done = strcmp(all_states,'finished') ;
            if sum(all_done) == length(jobs)
                keepChecking =  0 ;
                fprintf('all done\n');
                fig_alert
                toc(globalTic)
            end
            pause(1)
            % every five minutes, report status
            if ~mod(round(toc(globalTic)), 60*5)
                fprintf('still running, been %1.1f min for %d jobs \n',...
                    toc(globalTic)/60, length(jobs) ) ;
            end
        end
    end % of myFlags.keepChecking
    
    
end % of myFlags.runEvolutions


%%  GATHER RESULTS
if myFlags.collectEvolutions
    
    % create montage of all final evolutions
    dirinfo = dir(my_local_path);
    dirinfo(~[dirinfo.isdir]) = [];  %remove non-directories
    tf = ismember( {dirinfo.name}, {'.', '..'});
    dirinfo(tf) = [];  %remove current and parent directory.
    
    for iSub = 1:length(dirinfo)
        
        subdirinfo = dir( fullfile( dirinfo(iSub).folder,dirinfo(iSub).name ,'*mat') );
        all_evos = []  ;
        all_unitNames = [] ;
        for iEvo = 1:length(subdirinfo)
            
            tEvo_name = fullfile(subdirinfo(iEvo).folder, subdirinfo(iEvo).name ) ;
            tEvo = load(tEvo_name) ;
            areUnder =  regexp( subdirinfo(iEvo).name , '_' ) ;
            
            tEvo_layer = subdirinfo(iEvo).name(1:areUnder(1)-1) ;
            tEvo_chan = subdirinfo(iEvo).name(areUnder(1)+1:areUnder(2)-1)  ;
            
            try
                all_evos = cat(4,all_evos,tEvo.im.*tEvo.mask);
                all_unitNames = cat(1,all_unitNames,{sprintf('%s - %s',tEvo_layer, tEvo_chan ) });
                
            catch
                continue
            end
        end % of iEvo
        if isempty(all_evos), continue, end % wrong experiment
        
        figure
        ns = num_subplots(size(all_evos,4)) ;
        for iPic = 1:size(all_evos,4)
            subtightplot(ns(1),ns(2),iPic)
            imshow(all_evos(:,:,:,iPic))
            title(all_unitNames{iPic})
        end
        figPaper([12 12], my_final_path, sprintf('layer %s', tEvo_layer) );
        if iSub ==1
            if ~exist('net','var'); net = alexnet; end
            figure, montage(rescale(net.Layers(2).Weights))
            figPaper([12 12], my_final_path, sprintf('actual weights, alexnet, layer %s', tEvo_layer) );
        end
        
    end % of iUnit
end % of myFlags.collectEvolutions
