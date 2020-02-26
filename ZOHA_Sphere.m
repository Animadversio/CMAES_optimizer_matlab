classdef ZOHA_Sphere < handle
	properties
		dimen = space_dimen  % dimension of input space
        B = population_size  % population batch size
        mu = mu  % scale of the Gaussian distribution to estimate gradient
        assert Lambda > 0
        Lambda = Lambda  % diagonal regularizer for Hessian matrix
        lr = lr  % learning rate (step size) of moving along gradient
        sphere_norm = sphere_norm
        tang_codes = zeros((self.B, self.dimen))

        grad = zeros((1, dimen))  % estimated gradient
        innerU = zeros((B, dimen))  % inner random vectors with covariance matrix Id
        outerV = zeros((B, dimen))  % outer random vectors with covariance matrix H^{-1}, equals innerU @ H^{-1/2}
        xcur = zeros((1, dimen))  % current base point
        xnew = zeros((1, dimen))  % new base point
        Hupdate_freq = int(Hupdate_freq)  % Update Hessian (add additional samples every how many generations)
        HB = population_size  % Batch size of samples to estimate Hessian, can be different from B
        HinnerU = zeros((HB, dimen))  % sample deviation vectors for Hessian construction
        % SVD of the weighted HinnerU for Hessian construction
        HessUC = zeros((HB, dimen))  % Basis vector for the linear subspace defined by the samples
        HessD  = zeros(HB)  % diagonal values of the Lambda matrix
        HessV  = zeros((HB, HB))  % seems not used....
        HUDiag = zeros(HB)
        hess_comp = False
        _istep = 0  % step counter
        maximize = maximize  % maximize / minimize the function
        rankweight = rankweight % Switch between using raw score as weight VS use rank weight as score
        
    end

   	methods
   		function obj = ZOHA_Sphere(space_dimen, population_size=40, lr_norm=0.5, mu_norm=5, lr_sph=2, mu_sph=0.005,
            Lambda=1, Hupdate_freq=5, maximize=true, max_norm=300, rankweight=False, nat_grad=False)

            fprintf("Spereical Space dimension: %d, Population size: %d, Select size:%d, Optimization Parameters:\n Learning rate: %.3f\n"
               ,self.dimen, self.B, self.mu, self.lr)
   		end % of initialization


   		function [new_samples] =  doScoring(self,codes,scores,maximize,TrialRecord)
        N = self.dimen
        if self.hess_comp  % if this flag is true then more samples have been added to the trial
            throw;
            self.step_hessian(scores)
            % you should only get images for gradient estimation, get rid of the Hessian samples, or make use of it to estimate gradient
            codes = codes(:self.B+1, :)
            scores = scores(:self.B+1)
            self.hess_comp = false
        end

        if self._istep == 0
            % Population Initialization: if without initialization, the first xmean is evaluated from weighted average all the natural images
            fprintf('First generation\n')
            self.xcur = codes(1, :);
            self.xnew = codes(1, :);
            % No reweighting as there should be a single code
        else
            % self.xcur = self.xnew % should be same as following line
            self.xcur = codes(1, :);
            if self.rankweight == false % use the score difference as weight
                % B normalizer should go here larger cohort of codes gives more estimates 
                self.weights = (scores(2:) - scores(1) / self.B; % / self.mu 
            else  % use a function of rank as weight, not really gradient. 
                if self.maximize == false % note for weighted recombination, the maximization flag is here. 
                    [~,code_rank]=ismember(scores(2:) ,sort(scores(2:),'ascend')); % find rank of ascending order
                else
                    [~,code_rank]=ismember(scores(2:) ,sort(scores(2:),'descend')); % find rank of descending order 
                end
                % Consider do we need to consider the basis code and score here? Or no? 
                % Note the weights here are internally normalized s.t. sum up to 1, no need to normalize more. 
                self.weights = rankweight(length(scores)-1)[code_rank] % map the rank to the corresponding weight of recombination
            end
            % estimate gradient from the codes and scores
            % assume self.weights is a row vector 
            HAgrad = self.weights' * self.tang_codes
            fprintf("Estimated Gradient Norm %f\n",  norm(HAgrad))

            if (~self.maximize) and (~self.rankweight), mov_sign = -1, else, mov_sign = 1, end
            self.xnew = ExpMap(self.xcur, mov_sign * self.lr * HAgrad)
        end
        % Generate new sample by sampling from Gaussian distribution
        self.tang_codes = zeros(self.B, N);  % Tangent vectors of exploration
        new_samples = zeros(self.B + 1, N);
        self.innerU = randn(self.B, N);  % Isotropic gaussian distributions
        self.outerV = self.innerU / sqrt(self.Lambda) + ((self.innerU * self.HessUC') * self.HUDiag) * self.HessUC; % H^{-1/2}U
        new_samples(1, :) = self.xnew;
        self.tang_codes = self.mu * self.outerV; % m + sig * Normal(0,C)
        new_samples(2:end, ) = ExpMap(self.xnew, self.tang_codes); 
        if mod((self._istep + 1), self.Hupdate_freq) == 0:
            % add more samples to next batch for hessian computation
            self.hess_comp = true;
            self.HinnerU = randn(self.HB, N);
            H_pos_samples = self.xnew + self.mu * self.HinnerU;
            H_neg_samples = self.xnew - self.mu * self.HinnerU;
            new_samples = [new_samples; H_pos_samples; H_neg_samples];
        end
        self._istep = self._istep + 1;
        new_samples = new_samples / norm(new_samples, axis=1)[:, np.newaxis] * self.sphere_norm; 
        end