%% Evaluation Function
scorefun = @(x) 100*sum((x(:,1:end-1).^2 - x(:,2:end)).^2, 2) + sum((x(:,1:end-1)-1).^2, 2);
MAXIMIZE = 0;
%% Some constants for ploting
colorseq = brewermap(101, 'Set1');
theta = linspace(0,2*pi);
UNIT_CIRC = [cos(theta'), sin(theta')];
%%
[Xgrid, Ygrid] = meshgrid(-5:0.1:5, -5:0.1:5);
score_surf = scorefun([Xgrid(:), Ygrid(:)]);
score_surf = reshape(score_surf, size(Xgrid));
figure(1); clf;hold on 
contour(Xgrid, Ygrid, score_surf, 500) % value is too high to plot 
%contour(Xgrid, Ygrid, score_surf, 0:5:1000) % plot part of contours
%pause
%
codes = [4,2] + randn([4,2]);
TrialRecord=struct();
init_x = [];
optimizer = CMAES_constraint(codes, [], [[-5,-5];[5,5]]);
figure(1);hold on; axis equal
scatter(codes(:,1), codes(:,2), 16, colorseq(i,:),'filled')
for i=1:100
    scores = scorefun(codes);
    [new_codes, new_ids, TrialRecord] = optimizer.doScoring(codes, scores, MAXIMIZE, TrialRecord);
    disp('Center of Distribution')
    disp(optimizer.xmean)
    %disp(sort(scores'))
    ellipsoid = optimizer.xmean + optimizer.sigma * UNIT_CIRC * optimizer.A;
    plot(ellipsoid(:, 1), ellipsoid(:, 2), 'Color', colorseq(i,:))
    scatter(new_codes(:,1), new_codes(:,2), 16, colorseq(i,:),'filled')
    if ~isempty(TrialRecord.User.out_of_bnd_sample)
        scatter(TrialRecord.User.out_of_bnd_sample(:,1), TrialRecord.User.out_of_bnd_sample(:,2), 16, colorseq(i,:), 'x')
    end
    pause
    codes = new_codes;
end
%%
% function f = scorefun(x)
%     if size(x,1) < 2 error('dimension must be greater one'); end
%     f = 100*sum((x(1:end-1,:).^2 - x(2:end,:)).^2) + sum((x(1:end-1,:)-1).^2);
% end