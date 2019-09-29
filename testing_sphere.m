%% Evaluation Spherical Function
sphere_coord = @(x) [atan2(x(:,1),x(:,2)), asin(x(:,3)./sum(x.^2, 2)) - pi/2];
scorefun = @(x) -2 * cos(6 * x(:, 1)) + 2 * cos (12 * x(:, 2)) + (pi/12 - x(:, 2)) .^ 2 + 4 + 7/2 * x(:, 1).^2 ;
scorefun_3d = @(v) scorefun(sphere_coord(v));
MAXIMIZE = 0;
%% Some constants for ploting
colorseq = brewermap(101, 'Set1');
theta = linspace(0,2*pi);
UNIT_CIRC = [cos(theta'), sin(theta')];

%%
% [Xgrid, Ygrid] = meshgrid(-5:0.1:5, -5:0.1:5);
% score_surf = scorefun([Xgrid(:), Ygrid(:)]);
% score_surf = reshape(score_surf, size(Xgrid));
% figure(1); clf;hold on 
% contour(Xgrid, Ygrid, score_surf, 500) % value is too high to plot 
% %contour(Xgrid, Ygrid, score_surf, 0:5:1000) % plot part of contours
%%
[theta_grid, phi_grid] = meshgrid(-pi:0.1:pi, -pi/2:0.1:pi/2);
X = cos(theta_grid) .* cos(phi_grid);
Y = sin(theta_grid) .* cos(phi_grid);
Z = sin(phi_grid); 
points = [X(:), Y(:), Z(:)]; 
F = scorefun(sphere_coord(points));
K = convhull(points);
figure(1)
patch('vertices',points,'faces',K,'FaceVertexCData',F,...
        'FaceColor','interp','EdgeColor','none'); % draw it 'cdata',F,
axis equal tight vis3d % set axis
view(3) % set camera view to a 3D position
%pause
%%
init_code = [0,-1,0];
codes = [0,-1,0] + 0.2*randn([4,3]);
TrialRecord=struct();
optimizer = CMAES_Sphere(codes, init_code);
figure(1);hold on;axis equal
for i=1:100
    scores = scorefun(codes);
    [new_codes, new_ids, TrialRecord] = optimizer.doScoring(codes, scores, MAXIMIZE, TrialRecord);
    disp('Center of Distribution')
    disp(optimizer.xmean)
    disp(sort(scores'))
    scatter3(codes(:,1), codes(:,2), codes(:,3), 16, colorseq(i,:),'filled')
%     ellipsoid = optimizer.xmean + optimizer.sigma * UNIT_CIRC * optimizer.A;
%     plot(ellipsoid(:, 1), ellipsoid(:, 2), 'Color', colorseq(i,:))
    pause
    codes = new_codes;
end


%%
% function f = scorefun(x)
%     if size(x,1) < 2 error('dimension must be greater one'); end
%     f = 100*sum((x(1:end-1,:).^2 - x(2:end,:)).^2) + sum((x(1:end-1,:)-1).^2);
% end