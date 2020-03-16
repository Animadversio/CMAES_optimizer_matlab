addpath D:\Matlab_add_on\WUSTL_CHPC
c = parcluster;
c.AdditionalProperties.EmailAddress = 'binxu.wang@wustl.edu';
%c.AdditionalProperties.GpuCard = 'gpu-card-name';   
c.AdditionalProperties.GpusPerNode = 1;
c.AdditionalProperties.MemUsage = '4000';
c.AdditionalProperties.QueueName = 'dque';    
c.AdditionalProperties.WallTime = '01:00:00';
c.saveProfile
% Run a batch script, capturing the diary, adding a path to the workers
% and transferring some required files.
j = batch(myCluster, 'script1', ...
          'AdditionalPaths', '\\Shared\Project1\HelperFiles',...
          'AttachedFiles', {'script1helper1', 'script1helper2'});
j = c.batch(@parallel_example, 1, {}, 'Pool', 8, ...
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