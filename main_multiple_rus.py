import os
import subprocess
import argparse

# Define the list of optimizers and test problems
optimizers = ['EmpiricalUCB', 'UCB', 'WassersteinUCB', 'SBOKDE', 'DRBOKDE', 'MMD', 'MMD_Minimax_Approx', 'Stable']
test_problems = ['Ackley', 'ThreeHumpCamel', 'Hartmann', 'Modified_Branin', 'Continuous_Vendor', 'SixHumpCamel', 'portfolio_optimization', 'portfolio_normal_optimization']

# # Define other parameters
init_size = 2 # Initial number of points
running_rounds = 4 # Number of total points 
repeat = 1 # Number of times to repeat each experiment
start_seed = 100
device = 'cuda'
beta = 1.5

# Loop through each combination of optimizer and test problem
for optimizer in optimizers:
    for test_problem in test_problems:
        # Construct the command to run the main script with the appropriate arguments
        command = [
            'python', 'main.py',
            '--Optimizer', optimizer,
            '--TestProblem', test_problem,
            '--init_size', str(init_size),
            '--running_rounds', str(running_rounds),
            '--repeat', str(repeat),
            '--start_seed', str(start_seed),
            '--device', device,
            '--beta', str(beta)
        ]
        
        try:
            # Execute the command
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running optimizer {optimizer} on test problem {test_problem}: {e}")
