classdef ZOHA_Sphere < handle
	properties
		dimen   % dimension of input space
        B   % population batch size
        mu   % scale of the Gaussian distribution to estimate gradient
        Lambda   % diagonal regularizer for Hessian matrix
        lr  % learning rate (step size) of moving along gradient
        sphere_norm 
        tang_codes 

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
        maximize  % maximize / minimize the function
        rankweight % Switch between using raw score as weight VS use rank weight as score
    end

   	methods
   		function self = ZOHA_Sphere(space_dimen, population_size=40, select_cutoff=20, lr=2, mu=0.005, Lambda=1, Hupdate_freq=201, 
   			maximize=true, max_norm=300, rankweight=false, nat_grad=false)
            assert(Lambda > 0)
            self.dimen = space_dimen;  % dimension of input space
            self.B = population_size;  % population batch size
            self.select_cutoff = int(select_cutoff);
            self.select_cutoff = int(population_size / 2);
            self.mu = mu;  % scale of the Gaussian distribution to estimate gradient
            self.lr = lr;  % learning rate (step size) of moving along gradient
            self.sphere_norm = sphere_norm;
            self.Lambda = Lambda;  % diagonal regularizer for Hessian matrix
            self.maximize = maximize;  % maximize / minimize the function
            self.rankweight = rankweight; % Switch between using raw score as weight VS use rank weight as score

            self.tang_codes = zeros(self.B, self.dimen);
            self.grad = zeros(1, self.dimen);  % estimated gradient
            self.innerU = zeros(self.B, self.dimen);  % inner random vectors with covariance matrix Id
            self.outerV = zeros(self.B, self.dimen);  % outer random vectors with covariance matrix H^{-1}, equals innerU @ H^{-1/2}
            self.xcur = zeros(1, self.dimen); % current base point
            self.xnew = zeros(1, self.dimen); % new base point
            self.Hupdate_freq = int(Hupdate_freq);  % Update Hessian (add additional samples every how many generations)
            self.HB = population_size;  % Batch size of samples to estimate Hessian, can be different from B
            self.HinnerU = zeros(self.HB, self.dimen);  % sample deviation vectors for Hessian construction
            % SVD of the weighted HinnerU for Hessian construction
            self.HessUC = zeros(self.HB, self.dimen);  % Basis vector for the linear subspace defined by the samples
            self.HessD  = zeros(self.HB, 1);  % diagonal values of the Lambda matrix
            self.HessV  = zeros(self.HB, self.HB);  % seems not used....
            self.HUDiag = zeros(self.HB, 1);
            
        
            fprintf("Spereical Space dimension: %d, Population size: %d, Optimization Parameters:\n Exploration: %.3f\n Learning rate: %.3f\n"
               ,self.dimen, self.B, self.mu, self.lr)
            if self.rankweight
            	fprintf("Using rank weight, selection size: %d\n", self.select_cutoff)
            end
   		end % of initialization


   		function [new_samples] =  doScoring(self,codes,scores,maximize,TrialRecord)
        N = self.dimen
        if self.hess_comp  % if this flag is true then more samples have been added to the trial
            error("Not Implemented");
            % self.step_hessian(scores)
            % % you should only get images for gradient estimation, get rid of the Hessian samples, or make use of it to estimate gradient
            % codes = codes(1:self.B+1, :)
            % scores = scores(1:self.B+1)
            % self.hess_comp = false
        end

        if self.istep == -1
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
                self.weights = (scores(2:end) - scores(1) / self.B; % / self.mu 
            else  % use a function of rank as weight, not really gradient. 
                if self.maximize == false % note for weighted recombination, the maximization flag is here. 
                    [~,code_rank]=ismember(scores(2:end), sort(scores(2:end),'ascend')); % find rank of ascending order
                else
                    [~,code_rank]=ismember(scores(2:end), sort(scores(2:end),'descend')); % find rank of descending order 
                end
                % Consider do we need to consider the basis code and score here? Or no? 
                % Note the weights here are internally normalized s.t. sum up to 1, no need to normalize more. 
                raw_weights = rankweight(length(code_rank)); 
                self.weights = raw_weights(code_rank); % map the rank to the corresponding weight of recombination
            end
            % estimate gradient from the codes and scores
            % assume self.weights is a row vector 
            HAgrad = self.weights' * self.tang_codes;
            fprintf("Estimated Gradient Norm %f\n",  norm(HAgrad))

            if (~self.maximize) and (~self.rankweight), mov_sign = -1, else, mov_sign = 1, end
            self.xnew = ExpMap(self.xcur, mov_sign * self.lr * HAgrad);
        end
        % Generate new sample by sampling from Gaussian distribution
        self.tang_codes = zeros(self.B, N);  % Tangent vectors of exploration
        new_samples = zeros(self.B + 1, N);
        self.innerU = randn(self.B, N);  % Isotropic gaussian distributions
        self.outerV = self.innerU / sqrt(self.Lambda) + ((self.innerU * self.HessUC') * self.HUDiag) * self.HessUC; % H^{-1/2}U
        new_samples(1, :) = self.xnew;
        self.tang_codes = self.mu * self.outerV; % m + sig * Normal(0,C)
        new_samples(2:end, ) = ExpMap(self.xnew, self.tang_codes); 
        if mod((self.istep + 1), self.Hupdate_freq) == 0:
            % add more samples to next batch for hessian computation
            self.hess_comp = true;
            self.HinnerU = randn(self.HB, N);
            H_pos_samples = self.xnew + self.mu * self.HinnerU;
            H_neg_samples = self.xnew - self.mu * self.HinnerU;
            new_samples = [new_samples; H_pos_samples; H_neg_samples];
        end
        self.istep = self.istep + 1;
        new_samples = new_samples / norm(new_samples, axis=1)[:, np.newaxis] * self.sphere_norm; 
        end