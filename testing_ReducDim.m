%% Evaluation Function
scorefun = @(x) 100*sum((x(:,1:end-1).^2 - x(:,2:end)).^2, 2) + sum((x(:,1:end-1)-1).^2, 2);
MAXIMIZE = 0;
%% Some constants for ploting
colorseq = brewermap(101, 'Set1');
theta = linspace(0,2*pi);
UNIT_CIRC = [cos(theta'), sin(theta')];
%% 2d visualization
[Xgrid, Ygrid] = meshgrid(-5:0.1:5, -5:0.1:5);
score_surf = scorefun([Xgrid(:), Ygrid(:)]);
score_surf = reshape(score_surf, size(Xgrid));
figure(1); clf;hold on 
contour(Xgrid, Ygrid, score_surf, 500) % value is too high to plot 
%contour(Xgrid, Ygrid, score_surf, 0:5:1000) % plot part of contours
%pause
%%
codes = [2,3] + randn([4,2]);
TrialRecord=struct();
init_x = [];
optimizer = CMAES_ReducDim(codes, [], 2);
basis = optimizer.getBasis('rand');%[1 0;0 1]
figure(1);hold on;axis equal
for i=1:100
    scores = scorefun(codes);
    red_codes = codes * basis';
    [new_red_codes, new_ids, TrialRecord] = optimizer.doScoring(red_codes, scores, MAXIMIZE, TrialRecord);
    new_codes = new_red_codes * basis;
    disp('Center of Distribution')
    disp(optimizer.xmean * basis)
    %disp(sort(scores'))
    scatter(codes(:,1), codes(:,2), 16, colorseq(i,:),'filled')
    ellipsoid = (optimizer.xmean + optimizer.sigma * UNIT_CIRC * optimizer.A) * basis;
    plot(ellipsoid(:, 1), ellipsoid(:, 2), 'Color', colorseq(i,:))
    % pause
    codes = new_codes;
end
%% high d testing
codes = [1,2,2,2] + randn([10,4]);
TrialRecord=struct();
init_x = [];
optimizer = CMAES_simple(codes, []);
basis = eye(4);
optimizer = CMAES_ReducDim(codes, [], 2);
basis = optimizer.getBasis('rand');%'rand'
%figure(1);hold on;axis equal
for i=1:500
    scores = scorefun(codes);
    if i == 1
        red_codes = codes * basis';
    else
        red_codes = new_red_codes; % coordinate transform to subspace
    end
    [new_red_codes, new_ids, TrialRecord] = optimizer.doScoring(red_codes, scores, MAXIMIZE, TrialRecord);
    new_codes = new_red_codes * basis; % coordinate transform to outer space
    disp('Center of Distribution')
    disp(optimizer.xmean * basis)
    %disp(sort(scores'))
    %scatter(codes(:,1), codes(:,2), 16, colorseq(i,:),'filled')
    %ellipsoid = (optimizer.xmean + optimizer.sigma * UNIT_CIRC * optimizer.A) * basis;
    %plot(ellipsoid(:, 1), ellipsoid(:, 2), 'Color', colorseq(i,:))
    % pause
    codes = new_codes;
end
% %%
% function f = scorefun(x)
%     if size(x,1) < 2 error('dimension must be greater one'); end
%     f = 100*sum((x(1:end-1,:).^2 - x(2:end,:)).^2) + sum((x(1:end-1,:)-1).^2);
% end