classdef GA_classic < handle
    
    properties
        codes
        scores
        popsize
        kT_mul
        kT
        n_conserve
        mut_size
        mut_rate
        parental_skew
        
        maximize
        istep
        curr_sample_idc
        curr_sample_ids
        next_sample_idx
        opts
%         curr_samples
%         genealogy
        %
        %     TrialRecord.User.population_size = [];
        %     TrialRecord.User.init_x =        [] ;
        %     TrialRecord.User.Aupdate_freq =  10.0;
        %     TrialRecord.User.thread =        [];
        
        
    end % of properties
    
    
    methods
        
        function obj = GA_classic(init_codes, init_ids, options) 
            % 2nd arg should be [] if we do not use init_x parameter. 
            % if we want to tune this algorithm, we may need to send in a
            % structure containing some initial parameters? like `sigma`
            % object instantiation and parameter initialization 
            if nargin==2
                options = struct();
            end
            obj.popsize = 40;
            obj.kT_mul = 2;
            obj.n_conserve = 10;
            obj.mut_size = 0.75;
            obj.mut_rate = 0.25;
            obj.parental_skew = 0.75;
            if isempty(init_ids)
                obj.curr_sample_ids = compose("init%03d",1:size(init_codes,1));
            else
                obj.curr_sample_ids = init_ids;
            end
            obj.next_sample_idx = 1;
            obj.maximize = true;
            obj.istep = 0;
            
            obj.opts.popsize = obj.popsize;
            obj.opts.kT_mul = obj.kT_mul;
            obj.opts.n_conserve = obj.n_conserve;
            obj.opts.mut_size = obj.mut_size;
            obj.opts.mut_rate = obj.mut_rate;
            obj.opts.parental_skew = obj.parental_skew;
            obj.opts.maximize = obj.maximize;
        end % of initialization
        
        function [new_samples, new_ids, TrialRecord] =  doScoring(self,codes,scores,maximize,TrialRecord)
            assert(length(scores) == size(codes,1), ...
                sprintf('number of scores (%d) != population size (%d)', length(scores), size(codes,1)));
            new_size = self.popsize;  % this may != len(curr_samples) if it has been dynamically updated
            % instead of chaining the genealogy, alias it at every step
            curr_genealogy = self.curr_sample_ids;
            % deal with nan scores:
            nan_mask = isnan(scores);
            n_nans = int32(sum(nan_mask));
            valid_mask = ~nan_mask;
            n_valid = int32(sum(valid_mask));
            assert(n_nans == 0)  % discard the part dealing with nans, assume all are valid. 
            % if some images have scores
            valid_scores = scores(valid_mask);
            self.kT = max(std(valid_scores) * self.kT_mul, 1e-8);  % prevents underflow kT = 0
            fprintf('kT: %f\n', self.kT);
            %sort_order = np.argsort(valid_scores)[::-1]  % sort from high to low
            %valid_scores = valid_scores(sort_order)
            if maximize
            [valid_scores, sort_order] = sort(valid_scores, 'descend');
            fitness = exp((valid_scores - valid_scores(1)) / self.kT); 
            else
            [valid_scores, sort_order] = sort(valid_scores, 'ascend'); 
            fitness = exp(-(valid_scores - valid_scores(1)) / self.kT); 
            end
            % Note: if new_size is smaller than n_valid, low ranking images will be lost
            % thres_n_valid = min(n_valid, new_size);
            % keep_samples   = codes(valid_mask(sort_order(1:thres_n_valid)),:); % new_samples(1:thres_n_valid, :)
            % keep_genealogy = curr_genealogy(valid_mask(sort_order(1:thres_n_valid))); % new_genealogy(1:thres_n_valid) 
            keep_samples   = codes(sort_order,:); % new_samples(1:thres_n_valid, :)
            keep_genealogy = curr_genealogy(sort_order);
            
            
            % skips first n_conserve samples
            n_mate = new_size - self.n_conserve; % - n_nans
            [mate_samples, mate_genealogy] = mate(keep_samples, keep_genealogy,...
                    fitness, n_mate, self.parental_skew);
            [mut_samples, mut_genealogy] = mutate(mate_samples, mate_genealogy,...
                    self.mut_size, self.mut_rate);
            new_samples =   [keep_samples(1:self.n_conserve,:); mut_samples];
            new_genealogy = [keep_genealogy(1:self.n_conserve), mut_genealogy];
            
%         self.curr_samples = new_samples;
%         self.genealogy = new_genealogy;
        self.curr_sample_idc = self.next_sample_idx : self.next_sample_idx + new_size - 1; % cumulative id .
        self.next_sample_idx = self.next_sample_idx + new_size;
        self.curr_sample_ids = arrayfun(@(idx)compose("gen%03d_%06d", self.istep, idx), self.curr_sample_idc);
        new_ids = self.curr_sample_ids;
        self.istep = self.istep + 1;
        end % of do scoring
    end % of properties
end % of classdef


function [population_new, genealogy_new] = mutate(population, genealogy, mutation_size, mutation_rate) % random_generator
    % generate mask for gene mutation 
    do_mutate = rand(size(population)) < mutation_rate; 
    population_new = population; %.copy()
    mut_num = sum(do_mutate, 'all');
    % add gaussian noise with size mutation_size to selected genes
    population_new(do_mutate) = population_new(do_mutate) + mutation_size * randn(mut_num, 1);
    genealogy_new = arrayfun(@(s)compose("%s+mut", s), genealogy);
end

function [new_samples, new_genealogy] = mate(population, genealogy, fitness, new_size, skew)%random_generator
    % fitness > 0
    % clean data
    % default skew = 0.5
    assert(size(population, 1) == length(genealogy))
    assert(size(population, 1) == length(fitness))
    if max(fitness) == 0
        [~, idx] = max(fitness);
        fitness(idx) = 0.001;
    end
    if min(fitness) <= 0
        fitness(fitness <= 0) = min(fitness(fitness > 0));
    end
    % create a CDF based on fitness to draw probability from
    fitness_bins = cumsum(fitness);
    fitness_bins = fitness_bins / fitness_bins(end); 
    % a function that takes a (0,1) uniform distributed value and returns which
    % bin in CDF it falls
    selectbin = @(val) sum(val > fitness_bins) + 1;
    % https://stackoverflow.com/questions/13914066/generate-random-number-with-given-probability-matlab
    parent1s = arrayfun(selectbin, rand(1, new_size));
    parent2s = arrayfun(selectbin, rand(1, new_size));
    new_samples = zeros(new_size, size(population, 2)); 
    new_genealogy = repmat("", 1, new_size);
    for i = 1:new_size
    % generate uniform random in (0,1) and binarize with threshold skew
        parentage = rand(1, size(population, 2)) < skew;
    % select gene randomly from parents by the parentage mask
        new_samples(i,  parentage) = population(parent1s(i),  parentage);
        new_samples(i, ~parentage) = population(parent2s(i), ~parentage);
        new_genealogy(i) = compose("%s+%s", genealogy(parent1s(i)), genealogy(parent2s(i)));
    end 
end