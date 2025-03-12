import botorch
import torch
# from kdesbo.utils import generate_initial_data, sample_kde, qmc_sample_kde, update_edf, get_kernel_matrix
from .utils import generate_initial_data, sample_kde, update_edf, get_kernel_matrix, sample_context_for_L_calculation
from botorch.models.gp_regression import SingleTaskGP
import math
from botorch.optim.optimize import optimize_acqf
from .acquisition import KDE_UCB, Stable_UCB, MMD_UCB, MMD_Minimax_Approx_UCB, KDE_DRBO_UCB, Wasserstein_UCB
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls.sum_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.acquisition.analytic import UpperConfidenceBound, ExpectedImprovement
from botorch.generation.gen import gen_candidates_scipy, TGenCandidates, gen_candidates_torch


class BOTorchOptimizer:
    def __init__(self, problem, init_size=10, running_rounds=200, device=torch.device('cpu'), beta=1.5):
        self.problem = problem
        self.init_size = init_size
        self.running_rounds = running_rounds
        self.train_X, self.train_Y, self.contexts = generate_initial_data(self.problem, self.init_size)
        self.stochastic_Y = None
        self.beta = beta
        self.best_Y = torch.max(self.train_Y)
        self.best_stochastic_Y = None
        self.cumulative_reward = None   
        self.cumulative_stochastic_regret = None
        self.cumulative_stochastic_reward = None
        self.device=device
        self.bounds = self.problem.bounds.to(self.device)

    def get_model(self, context=False):
        X_contexts = self.train_X
        if context:
            X_contexts = torch.cat((X_contexts, self.contexts), dim=-1)
        mean = self.train_Y.mean()
        sigma = self.train_Y.std()
        Y = (self.train_Y-mean)/sigma
        model = SingleTaskGP(X_contexts, Y.reshape(-1, 1)).to(device=self.device)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

    def evaluate_new_candidates(self, candidates, i):
        candidates = candidates.cpu()
        new_Y, new_contexts = self.problem(candidates) # this calls the evaluate_true function via the forward function
        self.train_X = torch.cat((self.train_X, candidates), dim=0)
        self.train_Y = torch.cat((self.train_Y, new_Y), dim=0)
        self.contexts = torch.cat((self.contexts, new_contexts), dim=0)
        if self.best_Y < new_Y:
            self.best_Y = new_Y
        print(f"At running_rounds {i}, the best instantaneous value is :{self.best_Y}")
        if self.cumulative_reward is None:
            self.cumulative_reward = torch.sum(self.train_Y.reshape(-1,))
        else:
            self.cumulative_reward += new_Y[0]
        next_stochastic_Y = None
        if self.problem.can_calculate_stochastic:
            next_stochastic_Y = None
            if self.stochastic_Y is None:
                self.stochastic_Y = torch.zeros_like(self.train_Y)
                for i, x in enumerate(self.train_X):
                    next_stochastic_Y = self.problem.evaluate_stochastic(x.reshape(1, -1))
                    self.stochastic_Y[i] = next_stochastic_Y
                self.best_stochastic_Y = torch.max(self.stochastic_Y)
                self.cumulative_stochastic_regret = torch.sum(self.problem.max_stochastic -self.stochastic_Y)
                self.cumulative_stochastic_reward = torch.sum(self.stochastic_Y)
            else:
                next_stochastic_Y = self.problem.evaluate_stochastic(candidates.reshape(1, -1))
                self.stochastic_Y = torch.cat((self.stochastic_Y, next_stochastic_Y.reshape(1, )))
                if self.best_stochastic_Y < next_stochastic_Y:
                    self.best_stochastic_Y = next_stochastic_Y
                self.cumulative_stochastic_regret += self.problem.max_stochastic-next_stochastic_Y
                self.cumulative_stochastic_reward += next_stochastic_Y
            # wandb.log({f"Best_Stochastic_Value": self.best_stochastic_Y, f"Best_Value": self.best_Y,
            #            f"Cumulative Stochastic Reward": self.cumulative_stochastic_reward,
            #            f"Cumulative Stochastic Regret": self.cumulative_stochastic_regret,
            #            f"Cumulative Reward":self.cumulative_reward})
        # else:
        #     wandb.log({f"Best_Value": self.best_Y, f"Cumulative Reward":self.cumulative_reward})
        print(f"At running_rounds {i}, the best robust value is :{self.best_stochastic_Y}")
        print(f"Candidate :{candidates}, new_value :{new_Y}, next_exp_value: {next_stochastic_Y}")

    def run_opt(self):
        raise NotImplementedError

class UCB_Optimizer(BOTorchOptimizer):
    def __init__(self, problem, init_size=10, running_rounds=150, device=torch.device('cpu'), beta=1.5):
        super().__init__(problem, init_size, running_rounds, device=device, beta=beta)

    def run_opt(self):
        for i in range(self.init_size, self.running_rounds):
            model = self.get_model()
            ucb = UpperConfidenceBound(
                model=model,
                beta=self.beta,
            )
            #print(self.train_X.shape)
            candidates, _ = optimize_acqf(
                acq_function=ucb,
                bounds=self.bounds[:, :(self.problem.dim-self.problem.contexts_dim)],
                q=1,
                num_restarts=10,
                raw_samples=1024,
                gen_candidates=gen_candidates_torch,
                options={"batch_limit": 512}
            )
            self.evaluate_new_candidates(candidates.detach(), i)
        return self.train_X, self.train_Y, self.contexts, self.cumulative_stochastic_regret, self.best_Y, self.stochastic_Y

class Stable_Optimizer(BOTorchOptimizer):
    def __init__(self, problem, init_size=10, running_rounds=200, device=torch.device('cpu'), beta=1.5):
        super().__init__(problem, init_size, running_rounds, device=device, beta=beta)

    def run_opt(self):
        for i in range(self.init_size, self.running_rounds):
            model = self.get_model(context=True)
            stable_ucb = Stable_UCB(
                model=model,
                beta=self.beta,
                contexts_observed=self.contexts.to(device=self.device),
            )
            candidates, _ = optimize_acqf(
                acq_function=stable_ucb,
                bounds=self.bounds[:, :(self.problem.dim-self.problem.contexts_dim)],
                q=1,
                num_restarts=10,
                raw_samples=1024,
                gen_candidates=gen_candidates_torch,
                options={"batch_limit": 512}
            )
            self.evaluate_new_candidates(candidates.detach(), i)
        return self.train_X, self.train_Y, self.contexts, self.cumulative_stochastic_regret, self.best_Y, self.stochastic_Y


class MMD_Optimizer(BOTorchOptimizer):
    def __init__(self, problem, init_size=10, running_rounds=200, num_discretization=100, device=torch.device('cpu'), beta=1.5):
        super().__init__(problem, init_size, running_rounds, device=device, beta=beta)
        self.num_discretization = num_discretization

    def run_opt(self):
        discretized_contexts = None
        if self.problem.contexts_dim == 1:
            discretized_contexts = (torch.linspace(self.problem.bounds[0,-1], self.problem.bounds[1,-1],
                                                  self.num_discretization).reshape(-1, 1)).to(device=self.device)
        else:
            x = []
            disc_per_dim = math.ceil(math.pow(float(self.num_discretization), 1/self.problem.contexts_dim))
            for i in range(self.problem.contexts_dim):
                x.append(torch.linspace(self.problem.bounds[0, self.problem.dim-self.problem.contexts_dim+i],
                                     self.problem.bounds[1, self.problem.dim-self.problem.contexts_dim+i],
                                     disc_per_dim).reshape(-1))
            discretized_contexts = torch.stack(x, dim=-1).to(device=self.device)

        for i in range(self.init_size, self.running_rounds):
            # For discretization.
            distribution = update_edf(self.contexts.to(self.device), discretized_contexts)
            model = self.get_model(context=True)
            kernel_matrix = get_kernel_matrix(model, before_dim=self.problem.dim-self.problem.contexts_dim,
                                              contexts=discretized_contexts)
            mmd_ucb = MMD_UCB(
                model=model,
                distribution=distribution,
                discretized_contexts=discretized_contexts,
                kernel_matrix=kernel_matrix,
                dist=1/math.sqrt(i) * (2.0+math.sqrt(2.0*math.log(10))),
                beta=self.beta,
                # contexts_observed=self.contexts,
            )
            candidates, _ = optimize_acqf(
                acq_function=mmd_ucb,
                bounds=self.bounds[:, :(self.problem.dim-self.problem.contexts_dim)],
                q=1,
                num_restarts=5,
                raw_samples=64,
                gen_candidates=gen_candidates_torch,
                options={"batch_limit": 64}
            )
            self.evaluate_new_candidates(candidates.detach(), i)
        return self.train_X, self.train_Y, self.contexts, self.cumulative_stochastic_regret, self.best_Y, self.stochastic_Y
class MMD_Minimax_Approx_Optimizer(BOTorchOptimizer):
    def __init__(self, problem, init_size=10, running_rounds=200,
                 num_discretization=1024, device=torch.device('cpu'), beta=1.5):
        super().__init__(problem, init_size, running_rounds, device=device, beta=beta)
        self.num_discretization = num_discretization

    def run_opt(self):
        discretized_contexts = None
        if self.problem.contexts_dim == 1:
            discretized_contexts = (torch.linspace(self.problem.bounds[0,-1], self.problem.bounds[1,-1],
                                                  self.num_discretization).reshape(-1, 1)).to(device=self.device)
        else:
            x = []
            disc_per_dim = math.ceil(math.pow(float(self.num_discretization), 1/self.problem.contexts_dim))
            for i in range(self.problem.contexts_dim):
                x.append(torch.linspace(self.problem.bounds[0, self.problem.dim-self.problem.contexts_dim+i],
                                     self.problem.bounds[1, self.problem.dim-self.problem.contexts_dim+i],
                                     disc_per_dim).reshape(-1))
            discretized_contexts = torch.stack(x, dim=-1).to(device=self.device)

        for i in range(self.init_size, self.running_rounds):
            # For discretization.
            distribution = update_edf(self.contexts.to(device=self.device), discretized_contexts)
            model = self.get_model(context=True)
            kernel_matrix = get_kernel_matrix(model, before_dim=self.problem.dim-self.problem.contexts_dim,
                                              contexts=discretized_contexts)
            mmd_ucb = MMD_Minimax_Approx_UCB(
                model=model,
                distribution=distribution,
                discretized_contexts=discretized_contexts,
                kernel_matrix=kernel_matrix,
                dist=1/math.sqrt(i) * (2.0+math.sqrt(2.0*math.log(10))),
                beta=self.beta,
                # contexts_observed=self.contexts,
            )
            candidates, _ = optimize_acqf(
                acq_function=mmd_ucb,
                bounds=self.bounds[:, :(self.problem.dim-self.problem.contexts_dim)],
                q=1,
                num_restarts=10,
                raw_samples=1024,
                gen_candidates=gen_candidates_torch,
                options={"batch_limit": 512}
            )
            self.evaluate_new_candidates(candidates.detach(), i)
        return self.train_X, self.train_Y, self.contexts, self.cumulative_stochastic_regret, self.best_Y, self.stochastic_Y

class SBOKDE_Optimizer(BOTorchOptimizer):
    def __init__(self, problem, num_kde_samples=512, init_size=10,
                 running_rounds=200, device=torch.device('cpu'), beta=1.5):
        super().__init__(problem, init_size, running_rounds, device=device, beta=beta)
        self.num_kde_samples = num_kde_samples

    def run_opt(self):
        for i in range(self.init_size, self.running_rounds):
            model = self.get_model(context=True)
            kde_samples = sample_kde(self.contexts, self.num_kde_samples,
                                     self.problem.bounds[:, (self.problem.dim-self.problem.contexts_dim):]) # (num_kde_samples, contexts_dim)
            kde_samples = kde_samples.to(self.device)
            kde_ucb = KDE_UCB(
                model=model,
                beta=self.beta,
                kde_samples=kde_samples,
            )
            candidates, _ = optimize_acqf(
                acq_function=kde_ucb,
                bounds=self.bounds[:, :(self.problem.dim-self.problem.contexts_dim)],
                q=1,
                num_restarts=10,
                raw_samples=1024,
                gen_candidates=gen_candidates_torch,
                options={"batch_limit":512}
            )
            self.evaluate_new_candidates(candidates.detach(), i)
        return self.train_X, self.train_Y, self.contexts, self.cumulative_stochastic_regret, self.best_Y, self.stochastic_Y

class WassersteinUCB_Optimizer(BOTorchOptimizer):
    # same as the EmpiricalSBO, but with Wasserstein_UCB acquisition function
    def __init__(self, problem, init_size=10, running_rounds=200, device=torch.device('cpu'), beta=1.5):
        super().__init__(problem, init_size, running_rounds, device=device, beta=beta)
        num_context_samples_for_L = 100
        # sample contexts for L calculation, shape is (num_context_samples_for_L, contexts_dim)
        self.contexts_for_L = sample_context_for_L_calculation(num_context_samples_for_L, self.problem.bounds[:, (self.problem.dim-self.problem.contexts_dim):])
    def run_opt(self):
        for i in range(self.init_size, self.running_rounds):
            model = self.get_model(context=True)
            wasserstein_ucb = Wasserstein_UCB(
                model=model,
                beta=self.beta,
                kde_samples=self.contexts,
                contexts_samples_for_L=self.contexts_for_L,
                radius = 0.3/math.sqrt(i),
            )
            candidates, _ = optimize_acqf(
                acq_function=wasserstein_ucb,
                bounds=self.bounds[:, :(self.problem.dim-self.problem.contexts_dim)],
                q=1,
                num_restarts=10, # number of restarts of the optimization algorithm
                raw_samples=1024,
                gen_candidates=gen_candidates_torch,
                options={"batch_limit":512} 
            )
            self.evaluate_new_candidates(candidates.detach(), i)

        return self.train_X, self.train_Y, self.contexts, self.cumulative_stochastic_regret, self.best_Y, self.stochastic_Y

class EmpiricalUCB_Optimizer(BOTorchOptimizer):
    # same as WassersteinUCB_Optimizer, but with radius = 0
    def __init__(self, problem, init_size=10, running_rounds=200, device=torch.device('cpu'), beta=1.5):
        super().__init__(problem, init_size, running_rounds, device=device, beta=beta)
        num_context_samples_for_L = 1
        # sample contexts for L calculation, shape is (num_context_samples_for_L, contexts_dim)
        self.contexts_for_L = sample_context_for_L_calculation(num_context_samples_for_L, self.problem.bounds[:, (self.problem.dim-self.problem.contexts_dim):])
    def run_opt(self):
        for i in range(self.init_size, self.running_rounds):
            model = self.get_model(context=True)
            wasserstein_ucb = Wasserstein_UCB(
                model=model,
                beta=self.beta,
                kde_samples=self.contexts,
                contexts_samples_for_L=self.contexts_for_L,
                radius = 0,
            )
            candidates, _ = optimize_acqf(
                acq_function=wasserstein_ucb,
                bounds=self.bounds[:, :(self.problem.dim-self.problem.contexts_dim)],
                q=1,
                num_restarts=10,
                raw_samples=1024,
                gen_candidates=gen_candidates_torch,
                options={"batch_limit":512} 
            )
            self.evaluate_new_candidates(candidates.detach(), i)

        return self.train_X, self.train_Y, self.contexts, self.cumulative_stochastic_regret, self.best_Y, self.stochastic_Y
    

class DRBOKDE_Optimizer(BOTorchOptimizer):
    def __init__(self, problem, num_kde_samples=1024, init_size=10,
                 running_rounds=200, Quasi=False, device=torch.device('cpu'), beta=1.5):
        super().__init__(problem, init_size, running_rounds, device=device, beta=beta)
        self.num_kde_samples = num_kde_samples
        self.Quasi = Quasi

    def run_opt(self):
        for i in range(self.init_size, self.running_rounds):
            model = self.get_model(context=True)
            kde_samples = sample_kde(self.contexts, self.num_kde_samples,
                                         self.problem.bounds[:, (self.problem.dim-self.problem.contexts_dim):])
            kde_samples = kde_samples.to(self.device)
            kde_ucb = KDE_DRBO_UCB(
                model=model,
                beta=self.beta,
                distance=1/math.pow(i, 2/(4+self.problem.contexts_dim)), 
                kde_samples=kde_samples,
                context_bound=self.problem.bounds[:, (self.problem.dim-self.problem.contexts_dim):],
            )
            candidates, _ = optimize_acqf(
                acq_function=kde_ucb,
                bounds=self.bounds[:, :(self.problem.dim-self.problem.contexts_dim)],
                q=1,
                num_restarts=5,
                raw_samples=64,
                gen_candidates=gen_candidates_torch,
                options={"batch_limit": 64}
            )
            self.evaluate_new_candidates(candidates.detach(), i)
        return self.train_X, self.train_Y, self.contexts, self.cumulative_stochastic_regret, self.best_Y, self.stochastic_Y
