%% Testing RFOptim
net = alexnet;
pic_size = net.Layers(1).InputSize;
refImg = imread("fc8_02.jpg");% BestImg;
refImg = imread("conv4_02_100.jpg");
%%
%% Circle mask 
Width = 256; Cent = Width /2 ; Radius = 100; %Center_dist = 40; 
img = 0.5 * ones(Width);
bg = 0.5 * ones(Width);
sigma = 10;
figure(37);imshow(bg)
contour_h = drawellipse(gca,'Center',[128,128],'SemiAxes',[50,40], 'RotationAngle', 10, 'Color',[0,0,0], 'FaceAlpha', 0.9);
ellips_mask = createMask(contour_h);
blurry_free_mask = imgaussfilt(double(1-ellips_mask),sigma);
maskedImg = (1 - blurry_free_mask).*double(refImg)/255 + blurry_free_mask * 0.5;
imshow(maskedImg)
% imwrite(bg, 'blurryfreeellipsmask2.png', 'Alpha', blurry_free_mask);
% blurry_free_mask = imgaussfilt(double(ellips_mask),sigma);
% imwrite(bg, 'blurryfreeellipsblob2.png', 'Alpha', blurry_free_mask);
%%
options = struct("init_sigma", 5);
constraint = [0, 0, 0, 0, 0;
              256, 256, 128, 128, 180];
init_x = [128,128,70,70,90];
Optimizer = CMAES_constraint(5, init_x, constraint, options);
fign = 37;
if ~isempty(fign)
    h = figure(fign);
    h.Position = [210         276        1201         645];
else
    h = figure();
    h.Position = [210         276        1201         645];
end
%%
genes = init_x;
% unit = {"fc8", 2};
unit = {"conv4", 2, 100};
my_layer = unit{1} ; 
iChan = unit{2} ; 
if contains(my_layer,"fc")
	t_unit = 1 ;
else
	t_unit = unit{3};
end
% get activation size
act1 = activations(net,rand(pic_size),my_layer,'OutputAs','Channels');
[nrows,ncols,nchans] = size(act1) ;
[i,j] = ind2sub( [nrows ncols], t_unit ) ;
% if ~exist(fullfile(my_final_path, my_layer, sprintf('%02d',t_unit) ),'dir')
%     mkdir(fullfile(my_final_path, my_layer, sprintf('%02d',t_unit) ))
% end
codes_all = [];
scores_all = [];
generations = [];
mean_activation = nan(1,n_gen) ;
for iGen = 1:n_gen
    % generate pictures
    pics = createMaskImage(refImg, genes);
    % feed them into net
    pics = imresize( pics , [pic_size(1) pic_size(2)]);
    % get activations
    act1 = activations(net, 255*pics, my_layer,'OutputAs','Channels'); % there is intrinsic normalization at 1st layer
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
    meanPic  = createMaskImage(refImg, mean(genes, 1));
    imagesc(meanPic);
    axis image off
    subplot(2,2,4)
    cla
    [mxscore, mxidx]= max(act_unit);
    maxPic  = createMaskImage(refImg, genes(mxidx, :));
    imagesc(maxPic);
    if mxidx == 1
        title(sprintf("basis %s",num2str(mxscore)))
    else
        title(num2str(mxscore))
    end
    axis image off
    drawnow
	end
% 	if Save
%     image_name = sprintf('%s_%03d_%02d_%02d_%02d_%02d.jpg',my_layer,iChan,t_unit,nrows,ncols,iGen) ;
%     imwrite( meanPic ,  ...
%         fullfile(my_final_path, my_layer, sprintf('%02d',t_unit), image_name ) , 'jpg')
%     end
    genes = genes_new;
    disp(genes)
end % of iGen
%%
function maskImgs = createMaskImage(refImg, genes)
gene_num = size(genes, 1);
maskImgs = zeros([size(refImg),gene_num]);
bg = 0.5 * ones(size(refImg));
for gene_i = 1:gene_num
    figure(37);imshow(bg)
    Cent = genes(gene_i, 1:2);
    SemiAxes = genes(gene_i, 3:4);
    Orientation = genes(gene_i, 5);
    contour_h = drawellipse(gca,'Center',Cent,'SemiAxes',SemiAxes, 'RotationAngle', Orientation, 'Color',[0,0,0], 'FaceAlpha', 0.9);
    ellips_mask = createMask(contour_h);
    blurry_free_mask = imgaussfilt(double(1-ellips_mask), 10);
    maskedImg = (1 - blurry_free_mask).*double(refImg) /255 + blurry_free_mask * 0.5;
    maskImgs(:,:,:,gene_i) = maskedImg;
end
end

% function plot_ellipse(cx,cy,a,b,angle)
% %a: width in pixels
% %b: height in pixels
% %cx: horizontal center
% %cy: vertical center
% %angle: orientation ellipse in degrees
% %color: color code (e.g., 'r' or [0.4 0.5 0.1])
% angle=angle/180*pi;
% r=0:0.1:2*pi+0.1;
% p=[(a*cos(r))' (b*sin(r))'];
% alpha=[cos(angle) -sin(angle)
%        sin(angle) cos(angle)];
% p1=p*alpha;
% patch(cx+p1(:,1),cy+p1(:,2),[0,0,0],'EdgeColor',[0,0,0]);
% end