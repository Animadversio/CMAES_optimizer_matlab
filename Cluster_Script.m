addpath D:\Matlab_add_on\WUSTL_CHPC
c = parcluster;
c.AdditionalProperties.EmailAddress = 'binxu.wang@wustl.edu';
%c.AdditionalProperties.GpuCard = 'gpu-card-name';   
c.AdditionalProperties.GpusPerNode = 1;
c.AdditionalProperties.MemUsage = '4000';
c.AdditionalProperties.QueueName = 'dque';    
c.AdditionalProperties.WallTime = '01:00:00';
c.saveProfile