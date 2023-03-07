import sys
  
# adding src to the system path
sys.path.insert(0, '/Users/nabilahmed/RL-book/')

import numpy as np
import matplotlib.pyplot as plt
import random
import math
from typing import Iterable, Iterator, TypeVar, List, Sequence
import rl.markov_process as mp

import itertools
from rl.dynamic_programming import evaluate_mrp
from rl.approximate_dynamic_programming import extended_vf

from rl.distribution import Categorical
from rl.markov_process import * 
from rl.function_approx import Gradient, Dynamic, Tabular
from rl.approximate_dynamic_programming import ValueFunctionApprox
from rl.monte_carlo import mc_prediction
from rl.td import td_prediction

from rl.returns import returns

from operator import itemgetter

import copy

S = TypeVar('S')

def td_lambda_tabular(
    traces: Iterable[Iterable[mp.TransitionStep[S]]],
    discount_factor: float,
    lambd: float
) -> Iterator[Mapping[S, float]]:
    value_func_dict = {}
    eligibility_trace_dict = {}
    count_dict = {}
    yield copy.deepcopy(value_func_dict)
    for trace in traces:
        # Limit traces to 10000 steps
        trace = itertools.islice(trace, 10000)
        eligibility_trace_dict = {}
        for transition_step in trace:
            reward = transition_step.reward
            state = transition_step.state
            next_state = transition_step.next_state
            # Update the eligibility trace
            for k in eligibility_trace_dict.keys():
                eligibility_trace_dict[k] = discount_factor * lambd * eligibility_trace_dict[k]
            eligibility_trace_dict.setdefault(state, 0)
            eligibility_trace_dict[state] += 1
            # We use the 1/n tabular update rule, so counts are used in place of alpha
            count_dict.setdefault(state, 0)
            count_dict[state] += 1
            # Perform the Tabular TD(lambda) update
            value_func_dict.setdefault(state, 0)
            value_func_dict.setdefault(next_state, 0)
            value_func_dict[state] += ((1/count_dict[state]) * eligibility_trace_dict[state] * 
                                        (reward + discount_factor * value_func_dict[next_state] - 
                                         value_func_dict[state]))

            yield copy.deepcopy(value_func_dict)

def td_lambda_func_approx(
    traces: Iterable[Iterable[mp.TransitionStep[S]]],
    approx_0: ValueFunctionApprox[S],
    discount_factor: float,
    lambd: float
) -> Mapping[S, float]:
    func_approx: ValueFunctionApprox[S] = approx_0

    yield copy.deepcopy(func_approx)

    for trace in traces:
        eligibility_trace = Gradient(func_approx).zero()
        # Limit traces to 10000 steps
        trace = itertools.islice(trace, 10000)
        for transition_step in trace:
            reward = transition_step.reward
            state = transition_step.state
            next_state = transition_step.next_state

            # Update the eligibility trace
            eligibility_trace = (eligibility_trace * (discount_factor * lambd) +
                func_approx.objective_gradient(
                xy_vals_seq=[(state, reward + discount_factor * extended_vf(func_approx, next_state))],
                obj_deriv_out_fun=lambda x1, _: np.ones(len(x1))
            ))

            # Perform the TD(lambda) update to the func approx
            func_approx = func_approx.update_with_gradient(
                eligibility_trace * 
                -1 * (reward + discount_factor * extended_vf(func_approx, next_state) 
                 - func_approx(state))
            )

            yield copy.deepcopy(func_approx)

def tabular_mc_prediction(
    traces: Iterable[Iterable[mp.TransitionStep[S]]],
    γ: float,
    episode_length_tolerance: float = 1e-6,
) -> Iterator[Mapping[S, float]]:
    '''traces is a finite iterable'''
    episodes: Iterator[Iterator[mp.ReturnStep[S]]] = \
        (returns(trace, γ, episode_length_tolerance) for trace in traces)
    count_dict = {}
    mean_dict = {}
    yield copy.deepcopy(mean_dict)
    for i, ep in enumerate(episodes):
        for step in ep:
            if step.state not in mean_dict:
                mean_dict[step.state] = 0
                count_dict[step.state] = 0

            if count_dict[step.state] == 0:
                mean_dict[step.state] = step.return_
            else:
                mean_dict[step.state] += (step.return_ - mean_dict[step.state])/(count_dict[step.state]+1)
            count_dict[step.state] += 1
            yield copy.deepcopy(mean_dict)

def tabular_td_prediction(
    traces: Iterable[Iterable[mp.TransitionStep[S]]],
    γ: float,
    episode_length_tolerance: float = 1e-6,
    initial_value: float = 0.0
) -> Iterator[Mapping[S, float]]:
    '''traces is a finite iterable'''
    episodes: Iterator[Iterator[mp.ReturnStep[S]]] = \
        (returns(trace, γ, episode_length_tolerance) for trace in traces)
    count_dict = {}
    mean_dict = {}
    yield copy.deepcopy(mean_dict)
    for i, ep in enumerate(episodes):
        for step in ep:
            if step.state not in mean_dict:
                mean_dict[step.state] = 0
                count_dict[step.state] = 0

            observed_val = step.reward + γ * mean_dict.get(step.next_state, initial_value)
            if count_dict[step.state] == 0:
                mean_dict[step.state] = observed_val
            else:
                mean_dict[step.state] += (observed_val - mean_dict[step.state])/(count_dict[step.state]+1)
            count_dict[step.state] += 1
            yield copy.deepcopy(mean_dict)

discount_factor = 0.4
state_to_plot = 10
num_steps_to_plot = 800

# Create a simple FiniiteMarkovRewardProcess
transition_matrix = np.zeros((25,25))
for i in range(25):
    for j in range(max(0, i-5), min(25, i+5)):
        transition_matrix[i, j] = 1/11
# Create the starting distribution at NonTerminal state 0
start_dist = Categorical({NonTerminal(10): 1.0})
# Create the FiniteMarkovRewardProcess
transition_reward = {}
for i in range(25):
    dist_dict = {(j, 20-abs(j-12)): transition_matrix[i, j] for j in range(25)}
    transition_reward[i] = Categorical(dist_dict)

sl_game_reward = FiniteMarkovRewardProcess(transition_reward_map=transition_reward)

# Value function from dynamic programming
dp_iters = evaluate_mrp(mrp = sl_game_reward, gamma = discount_factor)
print("Solution from dynamic programming:")
dp_iters_evolution = [i for i in itertools.islice(dp_iters, 50)]
print(dp_iters_evolution[-1])

# Value function from closed form solution
closed_form_sol = sl_game_reward.get_value_function_vec(gamma=discount_factor)
print("Solution from closed form solution:")
print(closed_form_sol)

# Value function from TD(lambda) tabular
game_traces = sl_game_reward.reward_traces(start_state_distribution=start_dist)

td_50_lambda_tabular_generator = td_lambda_tabular(traces = game_traces, 
                                           discount_factor = discount_factor, lambd = 0.50)
td_50_lambda_tabular_result = [next(td_50_lambda_tabular_generator) for i in range(10000)]

td_25_lambda_tabular_generator = td_lambda_tabular(traces = game_traces, 
                                           discount_factor = discount_factor, lambd = 0.25)
td_25_lambda_tabular_result = [next(td_25_lambda_tabular_generator) for i in range(10000)]

td_75_lambda_tabular_generator = td_lambda_tabular(traces = game_traces, 
                                           discount_factor = discount_factor, lambd = 0.75)
td_75_lambda_tabular_result = [next(td_75_lambda_tabular_generator) for i in range(10000)]

val_vec = ((k.state, v) for k, v in td_50_lambda_tabular_result[-1].items())
val_vec = sorted(val_vec, key=itemgetter(0))
val_vec = [round(b,8) for a, b in val_vec]
print("Solution from TD(lambda=0.5) tabular:")
print(val_vec)

# Value function from TD(lambda) function approximation
starting_func = Tabular()
td_lambda_func_approx_dict = td_lambda_func_approx(traces = game_traces, approx_0=starting_func,
                                                  discount_factor = discount_factor, lambd = 0.5)
td_lambda_approx_result = [next(td_lambda_func_approx_dict) for i in range(10000)]
val_vec = ((k.state, v) for k, v in td_lambda_approx_result[-1].values_map.items())
val_vec = sorted(val_vec, key=itemgetter(0))
val_vec = [round(b,8) for a, b in val_vec]
print("Solution from TD(0.5) function approximation:")
print(val_vec)

# Value function from MC tabular
mc_starting_func = Tabular()
mc_tabular_dict = tabular_mc_prediction(
    traces = game_traces,
    γ = discount_factor
)
mc_tabular_result = [next(mc_tabular_dict) for i in range(10000)]
val_vec = ((k.state, v) for k, v in mc_tabular_result[-1].items())
val_vec = sorted(val_vec, key=itemgetter(0))
val_vec = [round(b,8) for a, b in val_vec]
print("Solution from MC tabular:")
print(val_vec)

# Value function from TD Tabular
td_starting_func = Tabular()
td_tabular_dict = tabular_td_prediction(
    traces=game_traces,
    γ = discount_factor
)
td_tabular_result = [next(td_tabular_dict) for i in range(10000)]
val_vec = ((k.state, v) for k, v in td_tabular_result[-1].items())
val_vec = sorted(val_vec, key=itemgetter(0))
val_vec = [round(b,8) for a, b in val_vec]
print("Solution from TD tabular:")
print(val_vec)

# Plot the sample traces
#plt.plot([i[state_to_plot] for i in dp_iters_evolution], label='Dynamic Programming')
plt.plot([closed_form_sol[state_to_plot] for _ in range(num_steps_to_plot)], label='Closed Form Solution')
plt.plot([i.get(NonTerminal(state_to_plot),1) for i in td_25_lambda_tabular_result[:num_steps_to_plot]], label=f'TD(0.25) Tabular')
plt.plot([i.get(NonTerminal(state_to_plot),1) for i in td_50_lambda_tabular_result[:num_steps_to_plot]], label=f'TD(0.50) Tabular')
plt.plot([i.get(NonTerminal(state_to_plot),1) for i in td_75_lambda_tabular_result[:num_steps_to_plot]], label=f'TD(0.75) Tabular')
plt.plot([i.get(NonTerminal(state_to_plot),1) for i in mc_tabular_result[:num_steps_to_plot]], label='MC Tabular')
plt.plot([i.get(NonTerminal(state_to_plot),1) for i in td_tabular_result[:num_steps_to_plot]], label='TD Tabular')
plt.xlabel('Transition Steps')
plt.ylabel('V(s)')
plt.title('Convergence for Simple MarkovRewardProcess (s = 10)')
plt.legend(loc='lower right')

plt.savefig("assignment6/convergence_rates.png")
plt.clf()


from rl.chapter10.prediction_utils import (mc_finite_prediction_learning_rate, 
                                           td_finite_prediction_learning_rate,
                                           td_lambda_finite_prediction_learning_rate)
from rl.returns import returns
from rl.distribution import Choose
from math import sqrt

def compare_td_and_td_lambda_and_mc(
    fmrp: FiniteMarkovRewardProcess[S],
    gamma: float,
    mc_episode_length_tol: float,
    num_episodes: int,
    learning_rates: Sequence[Tuple[float, float, float]],
    initial_vf_dict: Mapping[NonTerminal[S], float],
    plot_batch: int,
    plot_start: int,
    lambda_list: Sequence[float],
    lambda_plotmarkers: Sequence[str]
) -> None:
    true_vf: np.ndarray = fmrp.get_value_function_vec(gamma)
    states: Sequence[NonTerminal[S]] = fmrp.non_terminal_states
    colors: Sequence[str] = ['r', 'y', 'm', 'g', 'c', 'k', 'b']

    import matplotlib.pyplot as plt
    plt.figure(figsize=(11, 7))

    for k, (init_lr, half_life, exponent) in enumerate(learning_rates):
        mc_funcs_it: Iterator[ValueFunctionApprox[S]] = \
            mc_finite_prediction_learning_rate(
                fmrp=fmrp,
                gamma=gamma,
                episode_length_tolerance=mc_episode_length_tol,
                initial_learning_rate=init_lr,
                half_life=half_life,
                exponent=exponent,
                initial_vf_dict=initial_vf_dict
            )
        mc_errors = []
        batch_mc_errs = []
        for i, mc_f in enumerate(itertools.islice(mc_funcs_it, num_episodes)):
            batch_mc_errs.append(sqrt(sum(
                (mc_f(s) - true_vf[j]) ** 2 for j, s in enumerate(states)
            ) / len(states)))
            if i % plot_batch == plot_batch - 1:
                mc_errors.append(sum(batch_mc_errs) / plot_batch)
                batch_mc_errs = []
        mc_plot = mc_errors[plot_start:]
        label = f"MC InitRate={init_lr:.3f},HalfLife" + \
            f"={half_life:.0f},Exp={exponent:.1f}"
        plt.plot(
            range(len(mc_plot)),
            mc_plot,
            color=colors[k],
            linestyle='-',
            label=label
        )

    sample_episodes: int = 1000
    td_episode_length: int = int(round(sum(
        len(list(returns(
            trace=fmrp.simulate_reward(Choose(states)),
            γ=gamma,
            tolerance=mc_episode_length_tol
        ))) for _ in range(sample_episodes)
    ) / sample_episodes))

    for k, (init_lr, half_life, exponent) in enumerate(learning_rates):
        td_funcs_it: Iterator[ValueFunctionApprox[S]] = \
            td_finite_prediction_learning_rate(
                fmrp=fmrp,
                gamma=gamma,
                episode_length=td_episode_length,
                initial_learning_rate=init_lr,
                half_life=half_life,
                exponent=exponent,
                initial_vf_dict=initial_vf_dict
            )
        td_errors = []
        transitions_batch = plot_batch * td_episode_length
        batch_td_errs = []

        for i, td_f in enumerate(
                itertools.islice(td_funcs_it, num_episodes * td_episode_length)
        ):
            batch_td_errs.append(sqrt(sum(
                (td_f(s) - true_vf[j]) ** 2 for j, s in enumerate(states)
            ) / len(states)))
            if i % transitions_batch == transitions_batch - 1:
                td_errors.append(sum(batch_td_errs) / transitions_batch)
                batch_td_errs = []
        td_plot = td_errors[plot_start:]
        label = f"TD InitRate={init_lr:.3f},HalfLife" + \
            f"={half_life:.0f},Exp={exponent:.1f}"
        plt.plot(
            range(len(td_plot)),
            td_plot,
            color=colors[k],
            linestyle='--',
            label=label
        )
    

    # loop thru lambda values
    for z, l in enumerate(lambda_list):
        for k, (init_lr, half_life, exponent) in enumerate(learning_rates):
            td_lambda_funcs_it: Iterator[ValueFunctionApprox[S]] = \
                td_lambda_finite_prediction_learning_rate(
                    fmrp=fmrp,
                    gamma=gamma,
                    lambd=l,
                    episode_length=td_episode_length,
                    initial_learning_rate=init_lr,
                    half_life=half_life,
                    exponent=exponent,
                    initial_vf_dict=initial_vf_dict
                )
            td_lambda_errors = []
            transitions_batch = plot_batch * td_episode_length
            batch_td_lambda_errs = []

            for i, td_lambda_f in enumerate(
                    itertools.islice(
                        td_lambda_funcs_it,
                        num_episodes * td_episode_length
                    )
            ):
                batch_td_lambda_errs.append(sqrt(sum(
                    (td_lambda_f(s) - true_vf[j]) ** 2 for j, s in enumerate(states)
                ) / len(states)))
                if i % transitions_batch == transitions_batch - 1:
                    td_lambda_errors.append(
                        sum(batch_td_lambda_errs) / transitions_batch
                    )
                    batch_td_lambda_errs = []
            td_lambda_plot = td_lambda_errors[plot_start:]
            label = f"TD Lambda={l} InitRate={init_lr:.3f},HalfLife" + \
                f"={half_life:.0f},Exp={exponent:.1f}"
            plt.plot(
                range(len(td_lambda_plot)),
                td_lambda_plot,
                color=colors[k],
                linestyle=":",
                label=label,
                marker = lambda_plotmarkers[z],
                markersize = 3
            )

    plt.xlabel("Episode Batches", fontsize=20)
    plt.ylabel("Value Function RMSE", fontsize=20)
    plt.title(
        "RMSE of MC, TD, and TD(lambda) as function of episode batches",
        fontsize=25
    )
    plt.grid(True)
    plt.legend(fontsize=10, loc='upper right')
    plt.savefig("assignment6/convergence_rates_rmse.png")

compare_td_and_td_lambda_and_mc(
    fmrp=sl_game_reward,
    gamma=0.8,
    mc_episode_length_tol=1e-6,
    num_episodes=700,
    learning_rates=[(0.01, 1e8, 0.5), (0.05, 1e8, 0.5)],
    initial_vf_dict={s: 0.5 for s in sl_game_reward.non_terminal_states},
    plot_batch=7,
    plot_start=0,
    lambda_list=[0.25, 0.5, 0.75],
    lambda_plotmarkers = ["o", "^", "s"]
)




                                                    
                                                    

