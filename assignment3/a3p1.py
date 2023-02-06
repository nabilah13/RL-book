import operator
from typing import Mapping, Iterator, TypeVar, Tuple, Iterator, TypeVar

import numpy as np
from rl.approximate_dynamic_programming import (ValueFunctionApprox, 
                                                evaluate_mrp, 
                                                value_iteration, 
                                                value_iteration_finite, 
                                                extended_vf)

from rl.distribution import Categorical
from rl.dynamic_programming import policy_iteration, policy_iteration_result
from rl.function_approx import Dynamic, Tabular
from rl.iterate import iterate
from rl.markov_process import NonTerminal
from rl.markov_decision_process import (FiniteMarkovDecisionProcess,
                                        MarkovRewardProcess)
from rl.policy import FinitePolicy, Policy, DeterministicPolicy

from rl.iterate import converged

A = TypeVar('A')
S = TypeVar('S')
V = Mapping[NonTerminal[S], float]

# Create an MDP similar to the wage problem from previous homework
job_loss_probs = [0.2, 0.8]
discount_factor = 0.8
wage_vector = np.array([5.0, 16.0, 7.0], dtype=float)
offer_prob_vector = np.array([0.5, 0.5], dtype=float)

state_to_action_dict_dict = {}

decline_state_reward_dict = {(0, wage_vector[0]): 1}
accept_1_state_reward_dict = {(0, wage_vector[0]): offer_prob_vector[1], (1, wage_vector[0]): offer_prob_vector[0]}
accept_2_state_reward_dict = {(0, wage_vector[0]): offer_prob_vector[0], (2, wage_vector[0]): offer_prob_vector[1]}
accept_state_reward_dict = {(j, wage_vector[0]): offer_prob_vector[j-1] for j in range(1, 3)}
action_dict = {'reject_all': Categorical(decline_state_reward_dict), 
               'accept_1': Categorical(accept_1_state_reward_dict),
               'accept_2': Categorical(accept_2_state_reward_dict),
               'accept_all': Categorical(accept_state_reward_dict)}

state_to_action_dict_dict[0] = action_dict

for i in range(1, 3):
    decline_state_reward_dict = {(0, wage_vector[i]): job_loss_probs[0], (i, wage_vector[i]): job_loss_probs[1]}

    accept_1_state_reward_dict = {(j, wage_vector[i]): job_loss_probs[0] * offer_prob_vector[j-1] for j in range(1,3)}
    accept_1_state_reward_dict[(0, wage_vector[i])] = accept_1_state_reward_dict[(2, wage_vector[i])]
    accept_1_state_reward_dict[(2, wage_vector[i])] = 0
    if i==1: 
        accept_1_state_reward_dict.pop((2, wage_vector[i]))
    accept_1_state_reward_dict[(i, wage_vector[i])] += job_loss_probs[1]

    accept_2_state_reward_dict = {(j, wage_vector[i]): job_loss_probs[0] * offer_prob_vector[j-1] for j in range(1,3)}
    accept_2_state_reward_dict[(0, wage_vector[i])] = accept_2_state_reward_dict[(1, wage_vector[i])]
    accept_2_state_reward_dict[(1, wage_vector[i])] = 0
    if i==2:
        accept_2_state_reward_dict.pop((1, wage_vector[i]))
    accept_2_state_reward_dict[(i, wage_vector[i])] += job_loss_probs[1]

    accept_state_reward_dict = {(j, wage_vector[i]): job_loss_probs[0] * offer_prob_vector[j-1] for j in range(1,3)}
    accept_state_reward_dict[(i, wage_vector[i])] += job_loss_probs[1]

    action_dict = {'reject_all': Categorical(decline_state_reward_dict), 
                   'accept_1': Categorical(accept_1_state_reward_dict),
                   'accept_2': Categorical(accept_2_state_reward_dict), 
                   'accept_all': Categorical(accept_state_reward_dict)}
    state_to_action_dict_dict[i] = action_dict


# Create the Finite MDP
wage_mdp = FiniteMarkovDecisionProcess(state_to_action_dict_dict)

print(wage_mdp)

# Use the policy iteration function from dynamic_programming.py
policy_itr = policy_iteration_result(wage_mdp, discount_factor)
print("Answer from policy_iteration in dynamic_programming.py")
print(policy_itr)
    
vf_approx = Dynamic({s: 0.0 for s in wage_mdp.non_terminal_states})
vf_approx_tabular = Tabular(values_map={s: 0.0 for s in wage_mdp.non_terminal_states},
                            count_to_weight_func=lambda x:1)
nt_state_uniform_dist = Categorical({s: 1 for s in wage_mdp.non_terminal_states})

# Use the value iteration function from approximate_dynamic_programming.py
value_itr = converged(value_iteration(wage_mdp, discount_factor, vf_approx_tabular, nt_state_uniform_dist, 1),
                            done = lambda x,y: x.within(y, 1e-3)) 
print("Answer from value_iteration in approximate_dynamic_progamming.py")
print(value_itr)

# Use the value iteration finite function from approximate_dynamic_programming.py
value_finite_itr = converged(value_iteration_finite(wage_mdp, discount_factor, vf_approx), 
                                done = lambda x,y: x.within(y, 1e-3)) 
print("Answer from value_iteration_finite in approximate_dynamic_progamming.py")
print(value_finite_itr)

#### My approximate policy iteration ####
# Helper function to set initial policy
def pick_random_policy(mdp: FiniteMarkovDecisionProcess[S, A]) -> DeterministicPolicy[S, A]:
        def pick_action(s: S) -> A:
            a = list(mdp.actions(NonTerminal(s)))[0]
            return a
        return DeterministicPolicy(pick_action)

# Helper function to obtain the greedy policy
def greedy_policy_from_vf_approx(
    mdp: FiniteMarkovDecisionProcess[S, A],
    vf_approx: ValueFunctionApprox[S],
    gamma: float
) -> DeterministicPolicy[S, A]:
    def optimal_action(s: S) -> A:
        s_nonterminal = NonTerminal(s)
        a, _ = max([(a, mdp.mapping[s_nonterminal][a].expectation(
                lambda s_r: s_r[1] + gamma * extended_vf(vf_approx, s_r[0]))
                ) for a in mdp.actions(s_nonterminal)
                ], key=operator.itemgetter(1))
        return a
    return DeterministicPolicy(optimal_action)

# Define our custom approximate policy iteration function
def approximate_policy_iteration(
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float,
    nt_states_distribution,
    num_state_samples: int
) -> Iterator[Tuple[V[S], FinitePolicy[S, A]]]:
    def update(vf_policy: Tuple[ValueFunctionApprox[S], Policy[S, A]])\
            -> Tuple[ValueFunctionApprox[S], DeterministicPolicy[S, A]]:

        vf_approx, pi = vf_policy
        mrp: MarkovRewardProcess[S] = mdp.apply_policy(pi)
        policy_vf_approx_itr = converged(evaluate_mrp(mrp, gamma, vf_approx, 
                                        non_terminal_states_distribution = nt_states_distribution,
                                        num_state_samples=num_state_samples), lambda x,y: x.within(y, 1e-3))
        policy_vf_approx = policy_vf_approx_itr
        
        improved_pi: DeterministicPolicy[S, A] = greedy_policy_from_vf_approx(
            mdp,
            policy_vf_approx,
            gamma
        )

        return policy_vf_approx, improved_pi

    v_0 = Dynamic({s: 0.0 for s in mdp.non_terminal_states})
    pi_0: DeterministicPolicy[S, A] = pick_random_policy(mdp, )
    return iterate(update, (v_0, pi_0))


approx_policy_itr = converged(approximate_policy_iteration(wage_mdp, discount_factor, 
                Categorical({s: 1 for s in wage_mdp.non_terminal_states}), 1), lambda x,y: x[0].within(y[0], 1e-3))
print("Answer from custom approximate policy iteration")
approx_value_vec, optimal_policy_func = approx_policy_itr
print(approx_value_vec)
for s in wage_mdp.non_terminal_states:
    print(s)
    print(optimal_policy_func.act(s))

