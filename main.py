from wdrbo.optimizers import *
from Benchmark.Test_Function import *
import argparse
import botorch
import numpy as np
import random
import torch
import pickle as pkl
import time
from multiprocessing import Pool, cpu_count
import os

parser = argparse.ArgumentParser()

parser.add_argument('--Optimizer', default='SBOKDE', type=str) 
parser.add_argument('--TestProblem', default='Ackley', type=str)
parser.add_argument('--Minimization', action="store_true")
parser.add_argument('--init_size', default=5, type=int)
parser.add_argument('--running_rounds', default=7, type=int)
parser.add_argument('--repeat', default=2, type=int)
parser.add_argument('--start_seed', default=100, type=int)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--beta', default=1.5, type=float)
parser.add_argument('--DistShift', default=False, type=bool)

args = parser.parse_args()

if (args.Optimizer=='MMD' or args.Optimizer=='DRBOKDE') and args.device!='cpu':
    args.device = 'cpu'
    print(f"GPU of {args.Optimizer} is not supported. Set to cpu.")
print(args)
device = torch.device(args.device)
print(device)

# Create the Result directory if it doesn't exist
os.makedirs('./Result', exist_ok=True)

test_func = eval(args.TestProblem)(negate=not args.Minimization).to(dtype=torch.float64)
for i in range(args.repeat):
    start = time.time()
    seed = args.start_seed + i
    # check if the same experiment has been already done
    if os.path.exists(f'./Result/{args.Optimizer}_{args.TestProblem}_{seed}_{args.running_rounds}_{test_func.dim}_{test_func.mu}_{test_func.sigma}_{args.beta}_X.pkl'):
        print(f"Experiment {args.Optimizer}_{args.TestProblem}_{seed}_{args.running_rounds}_{test_func.dim}_{test_func.mu}_{test_func.sigma}_{args.beta} has been already done.")
        continue
    

    botorch.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    opt = eval(args.Optimizer+'_Optimizer')(test_func, running_rounds=args.running_rounds, init_size=args.init_size, device=device, beta=args.beta)
    X, Y, contexts, cumulative_stochastic_regret, best_Y, stochastic_Y = opt.run_opt()
    
    # Save results
    with open(f'./Result/{args.Optimizer}_{args.TestProblem}_{seed}_{args.running_rounds}_{test_func.dim}_{test_func.mu}_{test_func.sigma}_{args.beta}_X.pkl','wb') as f:
        pkl.dump(X.cpu().detach().numpy(), f)
    with open(f'./Result/{args.Optimizer}_{args.TestProblem}_{seed}_{args.running_rounds}_{test_func.dim}_{test_func.mu}_{test_func.sigma}_{args.beta}_Y.pkl','wb') as f:
        pkl.dump(Y.cpu().detach().numpy(), f)
    with open(f'./Result/{args.Optimizer}_{args.TestProblem}_{seed}_{args.running_rounds}_{test_func.dim}_{test_func.mu}_{test_func.sigma}_{args.beta}_contexts.pkl','wb') as f:
        pkl.dump(contexts.cpu().detach().numpy(), f)
    with open(f'./Result/{args.Optimizer}_{args.TestProblem}_{seed}_{args.running_rounds}_{test_func.dim}_{test_func.mu}_{test_func.sigma}_{args.beta}_stochastic_Y.pkl','wb') as f:
        if stochastic_Y is not None:
            pkl.dump(stochastic_Y.cpu().detach().numpy(), f)
    with open(f'./Result/{args.Optimizer}_{args.TestProblem}_{seed}_{args.running_rounds}_{test_func.dim}_{test_func.mu}_{test_func.sigma}_{args.beta}_cumulative_stochastic_regret.pkl','wb') as f:
        if cumulative_stochastic_regret is not None:
            pkl.dump(cumulative_stochastic_regret.cpu().detach().numpy(), f)
    with open(f'./Result/{args.Optimizer}_{args.TestProblem}_{seed}_{args.running_rounds}_{test_func.dim}_{test_func.mu}_{test_func.sigma}_{args.beta}_best_Y.pkl','wb') as f:
        if best_Y is not None:
            pkl.dump(best_Y.cpu().detach().numpy(), f)
    print(f"Time for {i}th run:{time.time() - start}")
    time_for_run = time.time() - start
    with open(f'./Result/{args.Optimizer}_{args.TestProblem}_{seed}_{args.running_rounds}_{test_func.dim}_{test_func.mu}_{test_func.sigma}_{args.beta}_time.pkl','wb') as f:
        pkl.dump(time_for_run, f)

    torch.cuda.empty_cache()

if __name__ == '__main__':
    seeds = [args.start_seed + i for i in range(args.repeat)]
    print("All done")