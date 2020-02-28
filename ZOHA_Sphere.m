classdef ZOHA_Sphere < handle
	properties
		dimen   % dimension of input space
        B   % population batch size
        mu   % scale of the Gaussian distribution to estimate gradient
        Lambda   % diagonal regularizer for Hessian matrix
        lr  % learning rate (step size) of moving along gradient
        sphere_norm 
        tang_codes 
        select_cutoff
        
        grad   % estimated gradient
        innerU   % inner random vectors with covariance matrix Id
        outerV   % outer random vectors with covariance matrix H^{-1}, equals innerU @ H^{-1/2}
        xcur   % current base point
        xnew   % new base point
        Hupdate_freq  % Update Hessian (add additional samples every how many generations)
        HB   % Batch size of samples to estimate Hessian, can be different from B
        HinnerU   % sample deviation vectors for Hessian construction
        % SVD of the weighted HinnerU for Hessian construction
        HessUC   % Basis vector for the linear subspace defined by the samples
        HessD    % diagonal values of the Lambda matrix
        HessV    % seems not used....
        HUDiag 
        hess_comp = false;
        istep = -1;  % step counter
        counteval = 0
        maximize  % maximize / minimize the function
        rankweight % Switch between using raw score as weight VS use rank weight as score
    end

   	methods
   		function self = ZOHA_Sphere(space_dimen, options)
            
            self.dimen = space_dimen;  % dimension of input space
            self.parseParameters(options); % parse the options into the initial values of optimizer
            
            self.tang_codes = zeros(self.B, self.dimen);
            self.grad = zeros(1, self.dimen);  % estimated gradient
            self.innerU = zeros(self.B, self.dimen);  % inner random vectors with covariance matrix Id
            self.outerV = zeros(self.B, self.dimen);  % outer random vectors with covariance matrix H^{-1}, equals innerU @ H^{-1/2}
            self.xcur = zeros(1, self.dimen); % current base point
            self.xnew = zeros(1, self.dimen); % new base point
            % self.Hupdate_freq = int(Hupdate_freq);  % Update Hessian (add additional samples every how many generations)
            self.HB = self.B;  % Batch size of samples to estimate Hessian, can be different from B
            self.HinnerU = zeros(self.HB, self.dimen);  % sample deviation vectors for Hessian construction
            % SVD of the weighted HinnerU for Hessian construction
            self.HessUC = zeros(self.HB, self.dimen);  % Basis vector for the linear subspace defined by the samples
            self.HessD  = zeros(self.HB, 1);  % diagonal values of the Lambda matrix
            self.HessV  = zeros(self.HB, self.HB);  % seems not used....
            self.HUDiag = zeros(self.HB, 1);
   		end % of initialization
        
        function parseParameters(self, opts)
        	if ~isfield(opts, "population_size"), opts.population_size = 40; end
    		if ~isfield(opts, "select_cutoff"), opts.select_cutoff = opts.population_size / 2; end
    		if ~isfield(opts, "mu"), opts.mu = 40; end
    		if ~isfield(opts, "lr"), opts.lr = 40; end
    		if ~isfield(opts, "sphere_norm"), opts.sphere_norm = 300; end
    		if ~isfield(opts, "Lambda"), opts.Lambda = 1; end
    		if ~isfield(opts, "maximize"), opts.maximize = true; end
    		if ~isfield(opts, "rankweight"), opts.rankweight = true; end
            if ~isfield(opts, "rankbasis"), opts.rankbasis = true; end
    		if ~isfield(opts, "Hupdate_freq"), opts.Hupdate_freq = 201; end
            self.B = opts.population_size;  % population batch size
            self.select_cutoff = floor(opts.select_cutoff);
            self.mu = opts.mu;  % scale of the Gaussian distribution to estimate gradient
            self.lr = opts.lr;  % learning rate (step size) of moving along gradient
            self.sphere_norm = opts.sphere_norm;
            self.Lambda = opts.Lambda;  % diagonal regularizer for Hessian matrix
            assert(self.Lambda > 0)
            self.maximize = opts.maximize;  % maximize / minimize the function
            self.rankweight = opts.rankweight; % Switch between using raw score as weight VS use rank weight as score
            self.rankbasis = opts.rankbasis; % whether include basis in the ranking comparison.
            self.Hupdate_freq = floor(opts.Hupdate_freq);  % Update Hessian (add additional samples every how many generations)
            fprintf("\nSpherical Space dimension: %d, Population size: %d, Optimization Parameters:\n Exploration: %.3f\n Learning rate: %.3f\n",...
               self.dimen, self.B, self.mu, self.lr)
            if self.rankweight
            	fprintf("Using rank weight, selection size: %d\n", self.select_cutoff)
            end
            % Parameter Checking 
            ExpectExplAng = (sqrt(self.dimen) * self.mu) / pi * 180; % Expected angular distance between sample and basis 
            fprintf("Expected angular exploration length %.1f deg\n",ExpectExplAng)
            if ExpectExplAng > 90
                warning("Estimated exploration range too large! Destined to fail! Check parameters!\n")
            end
            if self.rankweight
                weights = rankweight(self.B, self.select_cutoff);
                ExpectStepSize = sqrt(self.dimen * sum(weights.^2)) * self.mu * self.lr / pi * 180;
                % Expected angular distance of gradient step.  
                fprintf("Estimated angular step size %.1f deg\n", ExpectStepSize)
                if ExpectStepSize > 90 
	                warning("Estimated step size too large! Destined to fail! Check parameters!\n")
	            end
            end
        end
        
   		function [new_samples, new_ids] =  doScoring(self,codes,scores, maximize, TrialRecord)
        end
    end
end