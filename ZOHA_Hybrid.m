classdef ZOHA_Hybrid < handel
    """Gaussian Sampling method Hessian Aware Algorithm
    With a automatic switch between linear exploration and spherical exploration
    Our ultimate Optimizer. It's an automatic blend of Spherical class and normal class.
    TODO: Add automatic tuning of mu and step size according to the norm of the mean vector.
    """
    properties
           % def __init__(self, space_dimen, population_size=40, lr=0.1, mu=1, lr_sph=2, mu_sph=0.005, 
            %Lambda=0.9, Hupdate_freq=5, maximize=True, max_norm=300, rankweight=False, nat_grad=False):
        dimen   % dimension of input space
        B   % population batch size
        mu   % scale of the Gaussian distribution to estimate gradient
        assert Lambda > 0
        Lambda  % diagonal regularizer for Hessian matrix
        lr   % learning rate (step size) of moving along gradient
        grad %= np.zeros((1, self.dimen))  % estimated gradient
        lr_sph
        mu_sph

        innerU   % inner random vectors with covariance matrix Id
        outerV % outer random vectors with covariance matrix H^{-1}, equals self.innerU @ H^{-1/2}
        xnew   % new base point
        xscore = 0
        Hupdate_freq   % Update Hessian (add additional samples every how many generations)
        HB = population_size  % Batch size of samples to estimate Hessian, can be different from self.B
        HinnerU = np.zeros((self.HB, self.dimen))  % sample deviation vectors for Hessian construction
        % SVD of the weighted HinnerU for Hessian construction
        HessUC = np.zeros((self.HB, self.dimen))  % Basis vector for the linear subspace defined by the samples
        HessD  = np.zeros(self.HB)  % diagonal values of the Lambda matrix
        HessV  = np.zeros((self.HB, self.HB))  % seems not used....
        HUDiag = np.zeros(self.HB)
        hess_comp = False
        _istep = -1  % step counter
        maximize = maximize  % maximize / minimize the function
        tang_code_stored = np.array([]).reshape((0, dimen))
        score_stored = np.array([])
        N_in_samp = 0
        max_norm = max_norm
        % Options for `compute_grad` part
        rankweight = rankweight % Switch between using raw score as weight VS use rank weight as score
        nat_grad = nat_grad % use the natural gradient definition, or normal gradient. 
        sphere_flag = False % initialize the whole system as linear?

    end



    def new_generation(self, init_score, init_code):
        self.xscore = init_score
        self.score_stored = np.array([])
        self.xnew = init_code
        self.tang_code_stored = np.array([]).reshape((0, self.dimen))
        self.N_in_samp = 0
        self._istep += 1 

    def compute_hess(self, scores, Lambda_Frac=100):
        ''' Not implemented in spherical setting '''
        % fbasis = self.xscore
        % fpos = scores[:self.HB]
        % fneg = scores[-self.HB:]
        % weights = abs((fpos + fneg - 2 * fbasis) / 2 / self.mu ** 2 / self.HB)  % use abs to enforce positive definiteness
        % C = sqrt(weights[:, np.newaxis]) * self.HinnerU  % or the sqrt may not work.
        % % H = C^TC + Lambda * I
        % self.HessV, self.HessD, self.HessUC = np.linalg.svd(C, full_matrices=False)
        % self.Lambda = (self.HessD ** 2).sum() / Lambda_Frac
        % self.HUDiag = 1 / sqrt(self.HessD ** 2 + self.Lambda) - 1 / sqrt(self.Lambda)
        % print("Hessian Samples Spectrum", self.HessD)
        % print("Hessian Samples Full Power:%f \nLambda:%f" % ((self.HessD ** 2).sum(), self.Lambda) )

    def compute_grad(self, scores):
        % add the new scores to storage
        % refer to the original one, adapt from the HessAware_Gauss version. 
        self.score_stored = np.concatenate((self.score_stored, scores), axis=0) if self.score_stored.size else scores
        if self.rankweight is False: % use the score difference as weight
            % B normalizer should go here larger cohort of codes gives more estimates 
            self.weights = (self.score_stored - self.xscore) / self.score_stored.size % / self.mu 
            % assert(self.N_in_samp == self.score_stored.size)
        else:  % use a function of rank as weight, not really gradient. 
            % Note descent check **could be** built into ranking weight? 
            % If not better just don't give weights to that sample 
            if self.maximize is False: % note for weighted recombination, the maximization flag is here. 
                code_rank = np.argsort(np.argsort( self.score_stored))  % add - operator it will do maximization.
            else:
                code_rank = np.argsort(np.argsort(-self.score_stored))
            % Consider do we need to consider the basis code and score here? Or no? 
            % Note the weights here are internally normalized s.t. sum up to 1, no need to normalize more. 
            self.weights = rankweight(self.score_stored.size, mu=self.B / 2)[code_rank] % map the rank to the corresponding weight of recombination
            self.weights = self.weights[np.newaxis, :]
            % only keep the top 20 codes and recombine them.

        if self.nat_grad: % if or not using the Hessian to rescale the codes 
            % hagrad = self.weights @ (self.code_stored - self.xnew) % /self.mu
            hagrad = self.weights @ self.tang_code_stored % /self.mu
        else:
            Hdcode = self.Lambda * self.tang_code_stored + (
                    (self.tang_code_stored @ self.HessUC.T) * self.HessD **2) @ self.HessUC
            hagrad = self.weights @ Hdcode  % /self.mu

        print("Gradient Norm %.2f" % (np.linalg.norm(hagrad)))
        % if self.rankweight is False:
        %     if self.maximize:
        %         ynew = radial_proj(self.xnew + self.lr * hagrad, max_norm=self.max_norm)
        %     else:
        %         ynew = radial_proj(self.xnew - self.lr * hagrad, max_norm=self.max_norm)
        % else: % if using rankweight, then the maximization if performed in the recombination step. 
        %     ynew = radial_proj(self.xnew + self.lr * hagrad, max_norm=self.max_norm)
        % Summarize the conditional above to the following line.
        % If it's minimizing and it's not using rankweight, then travel inverse to the hagrad. 
        mov_sign = -1 if (not self.maximize) and (not self.rankweight) else 1 
        if not self.sphere_flag: 
            ynew = self.xnew + mov_sign * self.lr * hagrad % Linear space ExpMap reduced to just normal linear addition.
            if norm(ynew) > self.max_norm:
                print("Travelling outside the spherer\n Changing to Spherical mode at step %d",self._istep)
                self.sphere_flag = True 
                self.mu = self.mu_sph % changing the new parameters
                self.lr = self.lr_sph % changing the new parameters
                % maybe save the old ones
        else:  % Spherically move mean
            ynew = ExpMap(self.xnew, mov_sign * self.lr * hagrad)
            ynew = ynew / norm(ynew) * self.max_norm % exact normalization to the sphere.
        % ynew = radial_proj(ynew, max_norm=self.max_norm) % Projection 
        return ynew

    def generate_sample(self, samp_num=None, hess_comp=False):
        ''' Assume the 1st row of codes is the  xnew  new starting point '''
        N = self.dimen
        % Generate new sample by sampling from Gaussian distribution
        if hess_comp: 
            % Not implemented yet 
            % self.hess_comp = True
            self.HinnerU = randn(self.HB, N)
            H_pos_samples = self.xnew + self.mu * self.HinnerU
            H_neg_samples = self.xnew - self.mu * self.HinnerU
            new_samples = np.concatenate((H_pos_samples, H_neg_samples), axis=0)
            % new_samples = radial_proj(new_samples, self.max_norm)
        else:
            % new_samples = zeros((samp_num, N))
            self.innerU = randn(samp_num, N)  % Isotropic gaussian distributions
            self.outerV = self.innerU / sqrt(self.Lambda) + (
                        (self.innerU @ self.HessUC.T) * self.HUDiag) @ self.HessUC  % H^{-1/2}U
            tang_codes = self.mu * self.outerV % TODO: orthogonalize 
            if not self.sphere_flag: 
                new_samples = self.xnew + tang_codes % Linear space ExpMap reduced to just normal linear addition.
            else:  % Spherically move mean
                new_samples = ExpMap(self.xnew, tang_codes) % m + sig * Normal(0,C) self.mu *
            new_samples = radial_proj(new_samples, self.max_norm)
            self.tang_code_stored = np.concatenate((self.tang_code_stored, tang_codes), axis=0) if self.tang_code_stored.size else tang_codes  % only store the tangent codes.
            self.N_in_samp += samp_num
        return new_samples
        % set short name for everything to simplify equations