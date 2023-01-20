import matplotlib.pyplot as plt
import numpy as np

from rl.distribution import Categorical
from rl.markov_process import *

# Create transition matrix
# Start with uniform dist of moving 1 to 6 spaces forward
transition_matrix = np.zeros((101,101))
for i in range(100):
    for j in range(i+1, min(i+7, 101)):
        transition_matrix[i, j] = 1/6
# Prob of winning (term state) is 1 minus sum of probs on non-term states
transition_matrix[:,-1] = 1 - np.sum(transition_matrix[:, :-1], axis=1)
# Create mapping for the ladders and snakes
ls_dict = {1:38, 4:14, 8:10, 21:42, 
    28:76, 50:67, 71:92, 80:99,
    97:78, 95:56, 88:24, 62:18, 48:26, 
    36:6, 32:10}
# Edit transition matrix according to the mapping
for start_sq, end_sq in ls_dict.items():
    transition_matrix[:, end_sq] += transition_matrix[:, start_sq]
    transition_matrix[:, start_sq] = 0

# Create Transition object
# Create the transition mapping from ints to a dist over ints
transition = {}
for i in range(100):
    dist_dict = {j: transition_matrix[i, j] for j in range(100)}
    dist_dict[100] = transition_matrix[i, -1]
    transition[i] = Categorical(dist_dict)

# Create the FiniteMarkovProcess
sl_game = FiniteMarkovProcess(transition_map = transition)
# Create the starting distribution at NonTerminal state 0
start_dist = Categorical({NonTerminal(0): 1.0})
# Set up the game traces
game_traces = sl_game.traces(start_state_distribution=start_dist)
# Run 5 simulations to see the sample traces
sample_traces = [[s.state for s in next(game_traces)] for i in range(5)]

# Plot the sample traces
for trace in sample_traces:
    plt.plot(trace)
    plt.xlabel('Time Step')
    plt.ylabel('Square')
    plt.title('Snakes & Ladders MP Traces')

plt.savefig("assignment1/sample_traces.png")
plt.clf()

# Run 1000 simulations for the histogram of time steps to finish game
times_to_finish = [len(list(next(game_traces))) for i in range(1000)]

# Plot the histogram
plt.hist(times_to_finish, 50, density=True, alpha=0.75)
plt.xlabel('Time Steps to Finish')
plt.ylabel('Count Density')
plt.title('Snakes & Ladders MP Times to Finish')
plt.grid(True)
plt.savefig("assignment1/times_to_finish.png")
plt.clf()

# Create the FiniteMarkovRewardProcess
transition_reward = {}
for i in range(100):
    dist_dict = {(j, 1): transition_matrix[i, j] for j in range(100)}
    dist_dict[(100, 1)] = transition_matrix[i, -1]
    transition_reward[i] = Categorical(dist_dict)

sl_game_reward = FiniteMarkovRewardProcess(transition_reward_map=transition_reward)
expected_steps = sl_game_reward.get_value_function_vec(gamma=1.0)[0]
print(f"Expected rolls needed from starting to win: {expected_steps}")
