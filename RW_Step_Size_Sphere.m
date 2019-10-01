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