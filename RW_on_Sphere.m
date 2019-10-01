% Code to simulate the step size for one random walk on sphere! 
%

% sample_size = 7;
% sigma = 3;
% tic
% Angle_arr = zeros(10000,1);
% for i=1:10000
% TanVec = randn([sample_size,2]);
% x0 = [0,0,1];
% TanVec = [TanVec, zeros(sample_size, 1)];
% points = ExpMap(x0, sigma * TanVec);
% meanVec = mean(points, 1);
% cos_arc = dot(meanVec, x0)/norm(meanVec);
% Angle = acos(cos_arc);
% %disp(Angle)
% Angle_arr(i) = Angle;
% end
% toc
% figure(2); hold on 
% histogram(Angle_arr, 50)
%% Demo code for generating and comparing distribution
figure(3);clf;
sample_size = 7;
dimension = 4096;
sigma_arr = [0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2.0, 3.0];
for j =1:numel(sigma_arr)
sigma = sigma_arr(j);
tic
Angle_arr = zeros(10000,1);
for i=1:10000
    TanVec = randn([sample_size,dimension-1]);
    x0 = zeros(1, dimension);
    x0(end) = 1;
    TanVec = [TanVec, zeros(sample_size, 1)];
    points = ExpMap(x0, sigma * TanVec);
    meanVec = mean(points, 1);
    cos_arc = dot(meanVec, x0)/norm(meanVec);
    Angle = acos(cos_arc);
    %disp(Angle)
    Angle_arr(i) = Angle;
end
toc
Angle_col{j} = Angle_arr;
figure(3); hold on 
histogram(Angle_col{j}, 200,'BinLimits',[0,pi],'EdgeColor', 'none', 'FaceAlpha', 0.4)%stairs
end
legend(cellfun(@(x) num2str(x), num2cell(sigma_arr,1),'UniformOutput',false))
title(["Distribution of angle between a vector and the ExpMap ","of normal distibuted tangent vector on 2 sphere","(with different \sigma)"])
% figure(3); hold on 
% histogram(Angle_arr, 50)
%%
disp(RW_Step_Size_Sphere(40, 0.5, 3))
%%
function [Mean, Std] = RW_Step_Size_Sphere(sample_size, sigma, dimension)
    Trial_num = 1000;
    Angle_arr = zeros(Trial_num,1);
    for i=1:Trial_num
        TanVec = randn([sample_size, dimension-1]);
        x0 = zeros(1, dimension);
        x0(end) = 1;
        TanVec = [TanVec, zeros(sample_size, 1)];
        points = ExpMap(x0, sigma * TanVec);
        meanVec = mean(points, 1);
        cos_arc = dot(meanVec, x0)/norm(meanVec);
        Angle = acos(cos_arc);
        Angle_arr(i) = Angle;
    end
    Mean = mean(Angle_arr);
    Std = std(Angle_arr);
end
