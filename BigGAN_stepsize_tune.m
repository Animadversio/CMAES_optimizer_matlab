%% BigGAN space step size tune 
%  related to Evol_BigGAN_stepsize_tune.m but just in silico test of step
%  size.
G = torchBigGAN("biggan-deep-256");
% Embed_mat = G.get_embedding();
G = G.select_space("BigGAN"); % 
truncnorm = truncate(makedist("Normal"),-2,2);
%%
fig = figure(6);
axL = subplot('position',[0.02, 0.02, 0.45, 0.95]);
axR = subplot('position',[0.52, 0.02, 0.45, 0.95]);
%% Isotropic Gaussian
bsz = 40;
newmean = [truncnorm.random(1,128), 0.05*randn(1,128)];
newmeanimg = G.visualize(newmean);
sample_codes = newmean + [0.04 * randn(bsz, 128), 0.05 * randn(bsz, 128)];
img_samples = G.visualize(sample_codes);
% imshow(imtile(img_frame),'Parent',ax1)
% colorbar(ax1);caxis(ax1,score_sorted([select_n,1]))
% title(ax1,"Gen "+iGen+" selected images")
imshow(newmeanimg,'Parent',axL)
title(axL,"Visualize mean code")
imshow(imtile(img_samples),'Parent',axR)
title(axR,"Potential samples")
% imshow(imtile(imgs_next),'Parent',ax4)
% title(ax4,"Real samples / next gen "+(iGen+1))
%% Hessian Awared Gaussian
newmean = [truncnorm.random(1,128), 0.05*randn(1,128)];
newmeanimg = G.visualize(newmean);
sample_codes = newmean + [0.04 * randn(bsz, 128), 0.05 * randn(bsz, 128)];
img_samples = G.visualize(sample_codes);

imshow(newmeanimg,'Parent',axL)
title(axL,"Visualize mean code")
imshow(imtile(img_samples),'Parent',axR)
title(axR,"Potential samples")
%%
Hdata = py.numpy.load("E:\Cluster_Backup\BigGANH\summary\H_avg_1000cls.npz");
evc_cls = Hdata.get("eigvects_clas_avg").double;
eva_cls = Hdata.get("eigvals_clas_avg").double;
evc_nos = Hdata.get("eigvects_nois_avg").double;
eva_nos = Hdata.get("eigvals_nois_avg").double;
evc_all = Hdata.get("eigvects_avg").double;
eva_all = Hdata.get("eigvals_avg").double;
%% Exploration in noise x class space inverse scaled by Hessian spectrum 
bsz = 25;
newmean = [truncnorm.random(1,128), 0.05*randn(1,128)];
newmeanimg = G.visualize(newmean);
sample_dev = [0.3 * randn(bsz, 128) * evc_nos * diag(eva_nos.^(-1/5)) * evc_nos', ...
              0.1 * randn(bsz, 128) * evc_cls * diag(eva_cls.^(-1/5)) * evc_cls'];
sample_codes = newmean + sample_dev;
img_samples = G.visualize(sample_codes);
fprintf("Dev vector norm: noise: %.1f class: %.1f\n",mean(norm_axis(sample_dev(:,1:128),2)),mean(norm_axis(sample_dev(:,129:end),2)))
fprintf("New code vec norm: noise: %.1f class: %.1f\n",mean(norm_axis(sample_codes(:,1:128),2)),mean(norm_axis(sample_codes(:,129:end),2)))
imshow(newmeanimg,'Parent',axL)
title(axL,"Visualize mean code")
imshow(imtile(img_samples),'Parent',axR)
title(axR,"Potential samples")
%% Exploration in Whole space inverse scaled by Hessian spectrum 
bsz = 25;
newmean = [truncnorm.random(1,128), 0.05*randn(1,128)];
newmeanimg = G.visualize(newmean);
sample_dev = [0.2 * randn(bsz, 256) * evc_all * diag(eva_all.^(-1/6)) * evc_all'];
sample_codes = newmean + sample_dev;
img_samples = G.visualize(sample_codes);
fprintf("Dev vector norm: noise: %.1f class: %.1f\n",mean(norm_axis(sample_dev(:,1:128),2)),mean(norm_axis(sample_dev(:,129:end),2)))
fprintf("New code vec norm: noise: %.1f class: %.1f\n",mean(norm_axis(sample_codes(:,1:128),2)),mean(norm_axis(sample_codes(:,129:end),2)))
imshow(newmeanimg,'Parent',axL)
title(axL,"Visualize mean code")
imshow(imtile(img_samples),'Parent',axR)
title(axR,"Potential samples") 
%%
bsz = 25; weights = dummyOptim.weights; select_n=12;

newmean = [truncnorm.random(1,128), 0.05*randn(1,128)];
for i=1:10
newmeanimg = G.visualize(newmean);
sample_dev = [0.2 * randn(bsz, 256) * evc_all * diag(eva_all.^(-1/8)) * evc_all'];
sample_codes = newmean + sample_dev;
img_samples = G.visualize(sample_codes);
fprintf("Mean code vec norm: noise: %.1f class: %.1f\n",norm_axis(newmean(:,1:128),2),norm_axis(newmean(:,129:end),2))
fprintf("Dev vector norm: noise: %.1f class: %.1f\n",mean(norm_axis(sample_dev(:,1:128),2)),mean(norm_axis(sample_dev(:,129:end),2)))
fprintf("New code vec norm: noise: %.1f class: %.1f\n",mean(norm_axis(sample_codes(:,1:128),2)),mean(norm_axis(sample_codes(:,129:end),2)))
imshow(newmeanimg,'Parent',axL)
title(axL,"Visualize mean code")
imshow(imtile(img_samples),'Parent',axR)
title(axR,"Potential samples") 
code_idx = randperm(bsz,select_n);
newmean = weights * sample_codes(code_idx, :);
pause
end

%%

