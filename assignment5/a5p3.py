import itertools
import rl.markov_process as mp

from rl.chapter2.simple_inventory_mrp import SimpleInventoryMRPFinite
from rl.approximate_dynamic_programming import ValueFunctionApprox
from rl.distribution import Choose
from rl.function_approx import Tabular
from rl.monte_carlo import mc_prediction
from rl.returns import returns
from rl.td import td_prediction
from typing import Iterable, Mapping, TypeVar, Iterator


S = TypeVar('S')

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
        yield mean_dict

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
        yield mean_dict

if __name__ == "__main__":
    user_capacity = 2
    user_poisson_lambda = 1.0
    user_holding_cost = 1.0
    user_stockout_cost = 10.0

    user_gamma = 0.9

    si_mrp = SimpleInventoryMRPFinite(
        capacity=user_capacity,
        poisson_lambda=user_poisson_lambda,
        holding_cost=user_holding_cost,
        stockout_cost=user_stockout_cost
    )
    traces = si_mrp.reward_traces(Choose(si_mrp.non_terminal_states))

    tabular_mc_pred = tabular_mc_prediction(traces, γ=user_gamma)
    tabular_td_pred = tabular_td_prediction(traces, γ=user_gamma, initial_value=-30)

    tab_mc_func_approx = Tabular()
    tab_td_func_approx = Tabular()

    mc_pred = mc_prediction(traces, approx_0 = tab_mc_func_approx,γ=user_gamma)
    # Create iterable of transition steps for td_prediction
    episodes: Iterator[Iterator[mp.ReturnStep[S]]] = \
        (returns(trace, γ=user_gamma, tolerance = 1e-6) for trace in traces)
    transitions = [step for ep in itertools.islice(episodes, 1000) for step in ep]
    td_pred = td_prediction(transitions, approx_0 = tab_td_func_approx, γ=user_gamma)


    print("Default Tabular MC Prediction after 1000 steps:") 
    print([i for i in itertools.islice(mc_pred, 1000)][-1])
    print("Custom Tabular MC Prediction after 1000 steps:") 
    print([i for i in itertools.islice(tabular_mc_pred, 1000)][-1])


    print("Default Tabular TD Prediction after 1000 steps:") 
    print([i for i in itertools.islice(td_pred, 1000)][-1])
    print("Custom Tabular TD Prediction after 1000 steps:")
    print([i for i in itertools.islice(tabular_td_pred, 1000)][-1])