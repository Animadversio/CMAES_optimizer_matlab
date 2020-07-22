EmbedVects_mat = BGAN.get_embedding();
% rd_coord = run_umap(EmbedVects_mat','metric',"correlation");
%% statistics about the embedding center
Embed_norm = sqrt(sum(EmbedVects_mat.^2,1));
L2distmat = pdist(EmbedVects_mat','euclidean');
figure(12);
subtightplot(1,2,1,0.05)
hist(Embed_norm);title("Norm of Embedding vector")
subtightplot(1,2,2,0.05)
hist(L2distmat);title("L2 distance between 2 embedding vectors")
%% 
class_id = 374;2;
noise = truncnorm.random(1,128) * 0.7;
visualize_fun = @(embed) BGAN.visualize_latent([noise+zeros(size(embed,1),128),embed]);
%% norm scaling overall
embed = EmbedVects_mat(:,2)';
noise = truncnorm.random(1,128) * 0.7;
imgs = BGAN.visualize_latent(linspace(0.1,3,10)'*[noise, EmbedVects_mat(:,2)']);figure;montage(imgs)
%% scaling noise part
embed = EmbedVects_mat(:,2)';
noise = truncnorm.random(1,128) * 0.7;
imgs = BGAN.visualize_latent([linspace(0.1,3,10)'*noise, ones(10,1) * EmbedVects_mat(:,2)']);figure;montage(imgs)
%% scaling the class part
embed = EmbedVects_mat(:,2)';
noise = truncnorm.random(1,128) * 0.7;
imgs = BGAN.visualize_latent([ones(16,1)*noise, linspace(0.1,5,16)' * EmbedVects_mat(:,2)']);figure;montage(imgs)
%% Interpolation Among 2 classes 
N = 30;
% embed = EmbedVects_mat(:,2)';
noise = truncnorm.random(1,128) * 0.6; 
embedinterp = (linspace(-0.5,1.5,N)' * [1, -1] + [0,1]) * EmbedVects_mat(:, [2, 460])';
imgs = BGAN.visualize_latent([ones(N,1)*noise, embedinterp]);figure;montage(imgs)
distmat = pdist2(embedinterp,EmbedVects_mat','euclidean');
[Dsort,sortIdx] = sort(distmat, 2);
arrayfun(@(ID,D) ClassName(ID)+compose(" %.3f", D), sortIdx(:,1:4), Dsort(:, 1:4))
%% Find the Nearest Neigboring class center
distmat = pdist2(embedinterp,EmbedVects_mat','euclidean');
% [minD, minClassId] = min(distmat, [], 2);
[Dsort,sortIdx] = sort(distmat, 2);
arrayfun(@(ID) ClassName(ID), sortIdx(:,1:5));
arrayfun(@(ID,D) ClassName(ID)+compose(" %.3f", D), sortIdx(:,1:5), Dsort(:, 1:5))
%% 
ClassName = string(net.Layers(end).ClassNames);
%% Find the Nearest Neigboring class center
distmat = pdist2(embedinterp,EmbedVects_mat','euclidean');
% [minD, minClassId] = min(distmat, [], 2);
[Dsort,sortIdx] = sort(distmat, 2);
arrayfun(@(ID) ClassName(ID), sortIdx(:,1:5));
arrayfun(@(ID,D) ClassName(ID)+compose(" %.3f", D), sortIdx(:,1:5), Dsort(:, 1:5))
%%

