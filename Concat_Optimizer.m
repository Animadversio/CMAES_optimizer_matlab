classdef Concat_Optimizer < handle
    % wrapper to bundle 2 optimizers together. Each optimizer work on part of the
    % space.
    properties
        dimen
        dimen_list
        optimizer1
        optimizer2
        opts
    end

   	methods
   		function obj = Concat_Optimizer(space_dimen_list, optimizer1, optimizer2)
            obj.dimen = sum(space_dimen_list);
            obj.dimen_list = space_dimen_list;
            obj.optimizer1 = optimizer1;
            obj.optimizer2 = optimizer2;
            obj.opts = {optimizer1.opts, optimizer2.opts}; % save the options for optimizer
        end
        function [new_samples, new_ids] =  doScoring(obj, codes, scores, maximize, TrialRecord)

            codes_part1 = codes(:, 1:obj.dimen_list(1));
            codes_part2 = codes(:, obj.dimen_list(1)+1:end);
            [new_codes1, new_ids] =  obj.optimizer1.doScoring(codes_part1, scores, maximize, TrialRecord);
            [new_codes2,       ~] =  obj.optimizer2.doScoring(codes_part2, scores, maximize, TrialRecord);
            new_samples = [new_codes1, new_codes2]; % assume the population size is the same
        end
    end
end