classdef ZOHA_Cylind < handle
	properties
		dimen % dimension of input space
        B  % population batch size
        Lambda  % diagonal regularizer for Hessian matrix
        lr_norm  % learning rate (step size) of moving along radial direction
        mu_norm  % step size of exploration along the radial direction
        lr_sph   % learning rate (step size) of moving around the sphere 
        mu_sph   % step size of exploration around the sphere
        max_norm
        
        grad  % estimated gradient
        innerU % inner random vectors with covariance matrix Id
        outerV  % outer random vectors with covariance matrix H^{-1}, equals self.innerU @ H^{-1/2}
        tang_codes % stored tangent code vectors
        xcur   % current base point
        xnew   % new base point
        xscore
        code_norms
        
        % Hessian Update parameters
        Hupdate_freq  % Update Hessian (add additional samples every how many generations)
        HB % = population_size  % Batch size of samples to estimate Hessian, can be different from self.B
        HinnerU  % sample deviation vectors for Hessian construction
        % SVD of the weighted HinnerU for Hessian construction
        HessUC  % Basis vector for the linear subspace defined by the samples
        HessD   % diagonal values of the Lambda matrix
        HessV  % seems not used....
        HUDiag

        hess_comp = false; 
        istep  = -1;% step counter
        counteval = 0
        maximize  % maximize / minimize the function
        % Options for `compute_grad` part
        rankweight % Switch between using raw score as weight VS use rank weight as score
        rankbasis
        nat_grad  % use the natural gradient definition, or normal gradient.

        select_cutoff
    end

   	methods
   		function self = ZOHA_Cylind(space_dimen, options)
      % (space_dimen, population_size=40, lr_norm=0.5, mu_norm=5, lr_sph=2, mu_sph=0.005,
           % Lambda=1, Hupdate_freq=5, maximize=True, max_norm=300, rankweight=False, nat_grad=false)
        self.dimen = space_dimen;
        self.parseParameters(options); % parse the options into the initial values of optimizer
        
        self.grad = zeros(1, self.dimen);  % estimated gradient
        self.innerU = zeros(self.B, self.dimen);  % inner random vectors with covariance matrix Id
        self.outerV = zeros(self.B, self.dimen);  % outer random vectors with covariance matrix H^{-1}, equals self.innerU @ H^{-1/2}
        self.xcur = zeros(1, self.dimen);  % current base point
        self.xnew = zeros(1, self.dimen);  % new base point
        self.xscore = 0;
        % Hessian Update parameters
        self.HB = self.B;  % Batch size of samples to estimate Hessian, can be different from self.B
        self.HinnerU = zeros(self.HB, self.dimen);  % sample deviation vectors for Hessian construction
        % SVD of the weighted HinnerU for Hessian construction
        self.HessUC = zeros(self.HB, self.dimen);  % Basis vector for the linear subspace defined by the samples
        self.HessD  = zeros(self.HB);  % diagonal values of the Lambda matrix
        self.HessV  = zeros(self.HB, self.HB); % seems not used....
        self.HUDiag = zeros(self.HB);
        % self.tang_code_stored = reshape([], (0, self.dimen))
        % self.scores = []
        % self.N_in_samp = 0
   		end % of initialization
  
        function parseParameters(self, opts)
            if ~isfield(opts, "population_size"), opts.population_size = 40; end
            if ~isfield(opts, "select_cutoff"), opts.select_cutoff = opts.population_size / 2; end    
            if ~isfield(opts, "lr_norm"), opts.lr_norm = 40; end
            if ~isfield(opts, "mu_norm"), opts.mu_norm = 5; end
            if ~isfield(opts, "lr_sph"), opts.lr_sph = 2; end  
            if ~isfield(opts, "mu_sph"), opts.mu_sph = 0.005; end          
            if ~isfield(opts, "Lambda"), opts.Lambda = 1; end
            if ~isfield(opts, "Hupdate_freq"), opts.Hupdate_freq = 5; end
            if ~isfield(opts, "maximize"), opts.maximize = true; end
            if ~isfield(opts, "max_norm"), opts.max_norm = 300; end
            if ~isfield(opts, "rankweight"), opts.rankweight = false; end
            if ~isfield(opts, "rankbasis"), opts.rankbasis = false; end
            if ~isfield(opts, "nat_grad"), opts.nat_grad = false; end
    
            self.B = opts.population_size;  % population batch size
            self.select_cutoff = floor(opts.select_cutoff);
            self.lr_norm = opts.lr_norm;  % learning rate (step size) on radial direction
            self.mu_norm = opts.mu_norm;  % scale of the Gaussian distribution on radial direction
            self.lr_sph = opts.lr_sph; % learning rate (step size) of moving along gradient
            self.mu_sph = opts.mu_sph; % scale of the Gaussian distribution to estimate gradient
            self.Lambda = opts.Lambda;  % diagonal regularizer for Hessian matrix
            self.max_norm = opts.max_norm; % maximum norm to cap 

            assert(self.Lambda > 0)
            self.maximize = opts.maximize;  % maximize / minimize the function
            self.max_norm = opts.max_norm;
            self.rankweight = opts.rankweight; % Switch between using raw score as weight VS use rank weight as score
            self.rankbasis = opts.rankbasis; % Switch between using raw score as weight VS use rank weight as score
            self.nat_grad = opts.nat_grad; % use the natural gradient definition, or normal gradient.
            self.Hupdate_freq = floor(opts.Hupdate_freq);  % Update Hessian (add additional samples every how many generations)

            fprintf("\nCylindrical Space dimension: %d, Population size: %d, Optimization Parameters:\n Exploration: lr_norm: %.1f, mu_norm: %.4f, lr_sph: %.1f, mu_sph: %.4f, Lambda: %.2f",...
               self.dimen, self.B, self.lr_norm, self.mu_norm, self.lr_sph, self.mu_sph, self.Lambda)
            if self.rankweight
                fprintf("Using rank weight, selection size: %d\n", self.select_cutoff)
            end
            
            % Parameter Checking  Need to go back to logic and red throught agai
            ExpectExplAng = (sqrt(self.dimen) * self.mu_sph) / pi * 180; % Expected angular distance between sample and basis 
            fprintf("Expected angular exploration length %.1f deg\n",ExpectExplAng)
            if ExpectExplAng > 90
                warning("Estimated exploration range too large! Destined to fail! Check parameters!\n")
            end
            if self.rankweight
                weights = rankweight(self.B, self.select_cutoff);
                ExpectStepSize = sqrt(self.dimen * sum(weights.^2)) * self.mu_sph * self.lr_sph / pi * 180;
                % Expected angular distance of gradient step.  
                fprintf("Estimated angular step size %.1f deg\n", ExpectStepSize)
                if ExpectStepSize > 90 
                    warning("Estimated step size too large! Destined to fail! Check parameters!\n")
                end
            end
        end

        function [new_samples, new_ids] =  doScoring(self, codes, scores, maximize, TrialRecord)
        N = self.dimen;
%         if self.hess_comp  % if this flag is true then more samples have been added to the trial
%             error("Not Implemented");
%             % self.step_hessian(scores)
%             % % you should only get images for gradient estimation, get rid of the Hessian samples, or make use of it to estimate gradient
%             % codes = codes(1:self.B+1, :)
%             % scores = scores(1:self.B+1)
%             % self.hess_comp = false
%         end
        if self.istep == -1
            % Population Initialization: if without initialization, the first xmean is evaluated from weighted average all the natural images
            fprintf('First generation\n')
            self.xcur = codes(1, :);
            self.xnew = codes(1, :);
            normxnew = max(50, min(self.max_norm, norm(self.xcur)));
            % No reweighting as there should be a single code
        else
            % self.xcur = self.xnew % should be same as following line
            self.xcur = codes(1, :);
            if self.rankweight == false % use the score difference as weight
                % B normalizer should go here larger cohort of codes gives more estimates 
                weights = (scores(2:end) - scores(1)) / self.B; % / self.mu 
            else  % use a function of rank as weight, not really gradient. 
                
                if ~ self.rankbasis % if false, then exclude the first basis vector from rank (thus it receive no weights.)
                    rankedscore = scores(2:end);
                else
                    rankedscore = scores;
                end
                if self.maximize == false % note for weighted recombination, the maximization flag is here. 
                    [~,code_rank]=ismember(rankedscore, sort(rankedscore,'ascend')); % find rank of ascending order
                else
                    [~,code_rank]=ismember(rankedscore, sort(rankedscore,'descend')); % find rank of descending order 
                end
                % Note the weights here are internally normalized s.t. sum up to 1, no need to normalize more. 
                raw_weights = rankweight(length(code_rank)); 
                weights = raw_weights(code_rank); % map the rank to the corresponding weight of recombination
                % Consider the basis in our rank! but the weight will be wasted as we don't use it. 
                if self.rankbasis
                    weights = weights(2:end); % the weight of the basis vector will do nothing! as the deviation will be nothing
                end
            end
            % estimate gradient from the codes and scores
            % assume self.weights is a row vector 
            HAgrad = weights * self.tang_codes;
            normgrad = weights * (self.code_norms - norm(self.xcur))';  % Recombine norms to get,
            fprintf("Estimated Gradient Norm %f\n",  norm(HAgrad))
            fprintf("Estimated Radial Gradient Norm %f\n",  normgrad)
            % determine the moving direction and move. 
            if (~self.maximize) && (~self.rankweight), mov_sign = -1; else, mov_sign = 1; end
            self.xnew = ExpMap(self.xcur, mov_sign * self.lr_sph * HAgrad); % new basis vector
            normxnew = max(50,min(self.max_norm, norm(self.xcur) + mov_sign * self.lr_norm * normgrad)); % new norm of the basis  
        end
        % Generate new sample by sampling from Gaussian distribution
        self.tang_codes = zeros(self.B, N);  % Tangent vectors of exploration
        new_samples = zeros(self.B + 1, N);
        self.innerU = randn(self.B, N);  % Isotropic gaussian distributions
        self.outerV = self.innerU / sqrt(self.Lambda) + ((self.innerU * self.HessUC') .* self.HUDiag) * self.HessUC; % H^{-1/2}U
        self.outerV = self.outerV - (self.outerV * self.xnew') * self.xnew / norm(self.xnew)^2; % Orthogonalize 
        
        new_samples(1, :) = self.xnew;
        self.tang_codes = self.mu_sph * self.outerV; % m + sig * Normal(0,C)
        new_samples(2:end, :) = ExpMap(self.xnew, self.tang_codes); 
        new_norms = max(50, min(self.max_norm, normxnew + self.mu_norm * randn(1, self.B))); % new norm of the basis
        self.code_norms = new_norms;
        new_samples = renormalize(new_samples, [normxnew, new_norms]');
%         if mod((self.istep + 1), self.Hupdate_freq) == 0
%             % add more samples to next batch for hessian computation
%             self.hess_comp = true;
%             self.HinnerU = randn(self.HB, N);
%             H_pos_samples = self.xnew + self.mu * self.HinnerU;
%             H_neg_samples = self.xnew - self.mu * self.HinnerU;
%             new_samples = [new_samples; H_pos_samples; H_neg_samples];
%         end
        new_ids = [];
        for k = 1:size(new_samples,1)
            new_ids = [new_ids, sprintf("gen%03d_%06d", self.istep+1, self.counteval)];
            self.counteval = self.counteval + 1;
        end
        self.istep = self.istep + 1;
        end
    end

    end
