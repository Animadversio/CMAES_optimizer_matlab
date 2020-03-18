addpath D:\Matlab_add_on\WUSTL_CHPC
addpath D:\Matlab_add_on\WUSTL_CHPC\IntegrationScripts\chpc
configCluster;
c = parcluster;
c.AdditionalProperties.EmailAddress = 'binxu.wang@wustl.edu';
%c.AdditionalProperties.GpuCard = 'gpu-card-name';   
c.AdditionalProperties.GpusPerNode = 1;
c.AdditionalProperties.MemUsage = 4000;
c.AdditionalProperties.QueueName = 'dque';    
c.AdditionalProperties.WallTime = '01:00:00';
c.saveProfile
%%
% Run a batch script, capturing the diary, adding a path to the workers
% and transferring some required files.
j = batch(myCluster, 'script1', ...
          'AdditionalPaths', '\\Shared\Project1\HelperFiles',...
          'AttachedFiles', {'script1helper1', 'script1helper2'});
j = c.batch(@parallel_example, 1, {}, 'Pool', 1, ...
        'CurrentFolder','.', 'AutoAddClientPath',false);
% j = batch(cluster, fcn, N, {x1,..., xn}) 
%     runs the function specified by
%     a function handle or function name, fcn, on a worker using the
%     identified cluster.  The function returns j, a handle to the job object
%     that runs the function. The function is evaluated with the given
%     arguments, x1,...,xn, returning N output arguments.  The function file
%     for fcn is copied to the worker. 
wait(j)
diary(j,"mydiary.log")
id = j.ID;
clear j
j = c.findJob('ID', 4);
j.fetchOutputs{:};
%%
global net G 
net = alexnet;
pic_size = net.Layers(1).InputSize;
% declare your generator
% GANpath = "C:\Users\ponce\Documents\GitHub\Monkey_Visual_Experiment_Data_Processing\DNN";
GANpath = "D:\Github\Monkey_Visual_Experiment_Data_Processing\DNN";
addpath(GANpath)
G = FC6Generator('matlabGANfc6.mat');
%%
addpath geomtry_util\
unit = {"fc8",2};
n_gen = 100;nsr=0.2;
init_genes = normrnd(0,1,30,4096);
options = struct("population_size",40, "select_cutoff",20, "lr",1, "Lambda",1, "sphere_norm",250, ...
        "Hupdate_freq",201, "maximize",true, "rankweight",true, "rankbasis", false, "nat_grad",false,...
         "indegree", true, "mu_init", 50, "mu_final", 7.33);
Optimizer = ZOHA_Sphere_lr(4096, options);
Optimizer.lr_schedule(n_gen,"inv");
param_lab = "tmp";
out_path = "C:\Users\ponce\OneDrive - Washington University in St. Louis\Optimizer_Tuning\lrsched_test";
%[scores_all,codes_all,generations,norm_all,Optimizer,~] = GANoptim_test_fun(unit, n_gen, Optimizer, init_genes, nsr, [], param_lab, out_path);
%%
% j = batch(myCluster, 'GANoptim_test_fun', ...
%           'AdditionalPaths', '\\Shared\Project1\HelperFiles',...
%           'AttachedFiles', {'script1helper1', 'script1helper2'});
j = batch(c,'GANoptim_test_fun', 6, {unit, n_gen, Optimizer, init_genes, nsr, [], param_lab, out_path}, 'Pool', 1, ...
        'CurrentFolder','.');%, 'AutoAddClientPath',false
