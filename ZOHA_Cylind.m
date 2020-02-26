classdef ZOHA_Cylind < handle
	properties
		dimen # = space_dimen  # dimension of input space
        B  # = population_size  # population batch size
        assert Lambda > 0
        Lambda  # diagonal regularizer for Hessian matrix
        lr_norm  # learning rate (step size) of moving along gradient
        mu_norm  # scale of the Gaussian distribution to estimate gradient
        grad = zeros((1, obj.dimen))  # estimated gradient
        innerU = zeros((obj.B, obj.dimen))  # inner random vectors with covariance matrix Id
        outerV = zeros((obj.B, obj.dimen))  # outer random vectors with covariance matrix H^{-1}, equals self.innerU @ H^{-1/2}
        xnew = zeros((1, obj.dimen))  # new base point
        xscore = 0
        # Hessian Update parameters
        Hupdate_freq = int(Hupdate_freq)  # Update Hessian (add additional samples every how many generations)
        HB # = population_size  # Batch size of samples to estimate Hessian, can be different from self.B
        HinnerU = zeros((obj.HB, obj.dimen))  # sample deviation vectors for Hessian construction
        # SVD of the weighted HinnerU for Hessian construction
        HessUC = zeros((self.HB, self.dimen))  # Basis vector for the linear subspace defined by the samples
        self.HessD  = np.zeros(self.HB)  # diagonal values of the Lambda matrix
        self.HessV  = np.zeros((self.HB, self.HB))  # seems not used....
        self.HUDiag = np.zeros(self.HB)
        self.hess_comp = False

        self._istep = 0  # step counter
        self.maximize = maximize  # maximize / minimize the function
        self.tang_code_stored = np.array([]).reshape((0, self.dimen))
        self.score_stored = np.array([])
        self.N_in_samp = 0
        self.max_norm = max_norm
        # Options for `compute_grad` part
        self.rankweight = rankweight # Switch between using raw score as weight VS use rank weight as score
        self.nat_grad = nat_grad # use the natural gradient definition, or normal gradient.
        self.sphere_flag = True # initialize the whole system as linear?
        self.lr_sph = lr_sph
        self.mu_sph = mu_sph
    end

   	methods
   		function obj = ZOHA_Cylind(space_dimen, population_size=40, lr_norm=0.5, mu_norm=5, lr_sph=2, mu_sph=0.005,
            Lambda=1, Hupdate_freq=5, maximize=True, max_norm=300, rankweight=False, nat_grad=False)

   		end % of initialization


   		function [new_samples] =  doScoring(obj,codes,scores)

        end