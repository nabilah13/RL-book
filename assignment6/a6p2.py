from rl.monte_carlo import epsilon_greedy_policy, glie_mc_control
from rl.approximate_dynamic_programming import QValueFunctionApprox
from rl.approximate_dynamic_programming import NTStateDistribution
from rl.markov_decision_process import TransitionStep, FiniteMarkovDecisionProcess, NonTerminal
from rl.distribution import Choose, Gaussian, Constant
from rl.chapter11.control_utils import get_vf_and_policy_from_qvf
from rl.chapter3.simple_inventory_mdp_cap import SimpleInventoryMDPCap, InventoryState
from rl.function_approx import Tabular
from rl.dynamic_programming import value_iteration_result
from rl.chapter7.asset_alloc_discrete import AssetAllocDiscrete
from typing import Sequence, Callable, Tuple
import numpy as np
from rl.function_approx import DNNSpec, AdamGradient, DNNApprox, learning_rate_schedule
from rl.td import glie_sarsa, greedy_policy_from_qvf
from rl.iterate import converged
from rl.policy import FiniteDeterministicPolicy
from rl.approximate_dynamic_programming import evaluate_mrp
from operator import itemgetter
from pprint import pprint  




if __name__ == "__main__":
    # first starting with the inventory MDP problem. DP first
    capacity = 2
    poisson_lambda = 1.0
    holding_cost = 1.0
    stockout_cost = 10.0
    gamma=0.9

    si_mdp: FiniteMarkovDecisionProcess[InventoryState, int] =\
        SimpleInventoryMDPCap(
            capacity=capacity,
            poisson_lambda=poisson_lambda,
            holding_cost=holding_cost,
            stockout_cost=stockout_cost
        )


    opt_vf, opt_policy = value_iteration_result(si_mdp, gamma=gamma)
    print("DP Optimal Value Function:")
    print(opt_vf)
    print("DP Optimal Policy:")
    print(opt_policy)

    # now using MC control

    qval_0 = Tabular()
    mc_control = glie_mc_control(mdp=si_mdp, states=Choose(si_mdp.non_terminal_states),
                                 approx_0=qval_0, γ=gamma,
                                 ϵ_as_func_of_episodes=lambda x: 1/x)
    next(mc_control)

    def almost_equal_qvfs(
        v1,
        v2,
        tolerance: float = 1e-3
    ) -> bool:
        return max(abs(v1.values_map[s] - v2.values_map[s]) for s in v1.values_map) < tolerance

    opt_qvf = converged(
       mc_control,
        done=almost_equal_qvfs
    )
    vf, policy = get_vf_and_policy_from_qvf(si_mdp, opt_qvf)
    print(f"Tabular MC Control Value Function:")
    print(f"Value function: {vf}")
    print(f"Tabular MC Control Policy:")
    print(f"Policy: {policy}")

    # now using sarsa control

    qval_0 = Tabular(count_to_weight_func=learning_rate_schedule(0.10, 1000, 0.5))
    sarsa_control = glie_sarsa(mdp=si_mdp, states=Choose(si_mdp.non_terminal_states),
                                 approx_0=qval_0, γ=gamma,
                                    ϵ_as_func_of_episodes=lambda x: 1/x,
                                    max_episode_length=1000)
    for i in range(2000):
        q = next(sarsa_control)
    vf, policy = get_vf_and_policy_from_qvf(si_mdp, q)
    print(f"Tabular Sarsa Control Value Function:")
    print(f"Value function: {vf}")
    print(f"Tabular Sarsa Control Policy:")
    print(f"Policy: {policy}")



    ###
    # now using the asset allocation problem
    # starting with DP
    steps: int = 4
    μ: float = 0.13
    σ: float = 0.2
    r: float = 0.07
    a: float = 1.0
    init_wealth: float = 1.0
    init_wealth_stdev: float = 0.1

    excess: float = μ - r
    var: float = σ * σ
    base_alloc: float = excess / (a * var)

    risky_ret: Sequence[Gaussian] = [Gaussian(μ=μ, σ=σ) for _ in range(steps)]
    riskless_ret: Sequence[float] = [r for _ in range(steps)]
    utility_function: Callable[[float], float] = lambda x: - np.exp(-a * x) / a
    alloc_choices: Sequence[float] = np.linspace(
        2 / 3 * base_alloc,
        4 / 3 * base_alloc,
        11
    )
    feature_funcs: Sequence[Callable[[Tuple[float, float]], float]] = \
        [
            lambda _: 1.,
            lambda w_x: w_x[0],
            lambda w_x: w_x[1],
            lambda w_x: w_x[1] * w_x[1]
        ]
    dnn: DNNSpec = DNNSpec(
        neurons=[],
        bias=False,
        hidden_activation=lambda x: x,
        hidden_activation_deriv=lambda y: np.ones_like(y),
        output_activation=lambda x: - np.sign(a) * np.exp(-x),
        output_activation_deriv=lambda y: -y
    )
    init_wealth_distr: Gaussian = Gaussian(μ=init_wealth, σ=init_wealth_stdev)

    aad: AssetAllocDiscrete = AssetAllocDiscrete(
        risky_return_distributions=risky_ret,
        riskless_returns=riskless_ret,
        utility_func=utility_function,
        risky_alloc_choices=alloc_choices,
        feature_functions=feature_funcs,
        dnn_spec=dnn,
        initial_wealth_distribution=init_wealth_distr
    )

    it_qvf: Iterator[QValueFunctionApprox[float, float]] = \
        aad.backward_induction_qvf()

    print("Backward Induction on Q-Value Function")
    print("--------------------------------------")
    print()
    for t, q in enumerate(it_qvf):
        print(f"Time {t:d}")
        print()
        opt_alloc: float = max(
            ((q((NonTerminal(init_wealth), ac)), ac) for ac in alloc_choices),
            key=itemgetter(0)
        )[1]
        val: float = max(q((NonTerminal(init_wealth), ac))
                         for ac in alloc_choices)
        print(f"Opt Risky Allocation = {opt_alloc:.3f}, Opt Val = {val:.3f}")
        print("Optimal Weights below:")
        for wts in q.weights:
            pprint(wts.weights)
        print()

    # Now analytic solution
    print("Analytical Solution")
    print("-------------------")
    print()

    for t in range(steps):
        print(f"Time {t:d}")
        print()
        left: int = steps - t
        growth: float = (1 + r) ** (left - 1)
        alloc: float = base_alloc / growth
        vval: float = - np.exp(- excess * excess * left / (2 * var)
                               - a * growth * (1 + r) * init_wealth) / a
        bias_wt: float = excess * excess * (left - 1) / (2 * var) + \
            np.log(np.abs(a))
        w_t_wt: float = a * growth * (1 + r)
        x_t_wt: float = a * excess * growth
        x_t2_wt: float = - var * (a * growth) ** 2 / 2

        print(f"Opt Risky Allocation = {alloc:.3f}, Opt Val = {vval:.3f}")
        print(f"Bias Weight = {bias_wt:.3f}")
        print(f"W_t Weight = {w_t_wt:.3f}")
        print(f"x_t Weight = {x_t_wt:.3f}")
        print(f"x_t^2 Weight = {x_t2_wt:.3f}")
        print()

    # glie mc control for the asset allocation problem
    fa = aad.get_qvf_func_approx()
    for t in range(steps):
        print(f"Time {t:d}")
        mc_control = glie_mc_control(mdp=aad.get_mdp(t), states=aad.get_states_distribution(t),
                                    approx_0=fa, γ=gamma,
                                    ϵ_as_func_of_episodes=lambda x: 1/x)
        for i in range(2000):
            q = next(mc_control)
        fa = q

        opt_alloc: float = max(
                ((q((NonTerminal(init_wealth), ac)), ac) for ac in alloc_choices),
                key=itemgetter(0)
            )[1]
        val =max(q((NonTerminal(init_wealth), ac)) for ac in alloc_choices)
        print("Optimal Weights below:")
        for wts in q.weights:
            print(wts.weights)
        print(f"MC Control Optimal Risky Allocation  = {opt_alloc:.3f}, Opt Val = {val:.3f}")
    # det_pol = greedy_policy_from_qvf(q, aad.get_mdp(0).actions)
    # max(q((NonTerminal(init_wealth), ac)) for ac in alloc_choices)
    # print(f"MC Control Deterministic Policy:")
    # print(f"Policy: {det_pol}")
    # print(f"MC Control Value Function:")
    # print(f"Value function: {qvf_from_q_and_mdp(q, aad.get_mdp(0))}")

    # glie sarsa for the asset allocation problem
    fa = aad.get_qvf_func_approx()
    for t in range(steps):
        print(f"Time {t:d}")
        print()
        sarsa_control = glie_sarsa(mdp=aad.get_mdp(t), states=aad.get_states_distribution(t),
                                    approx_0=fa, γ=gamma,
                                    ϵ_as_func_of_episodes=lambda x: 1/x, max_episode_length=1000)
        for i in range(2000):
            q = next(sarsa_control)
        fa = q
        opt_alloc: float = max( ((q((NonTerminal(init_wealth), ac)), ac) for ac in alloc_choices), key=itemgetter(0))[1]
        val =max(q((NonTerminal(init_wealth), ac)) for ac in alloc_choices)
        print(f"SARSA Control Optimal Risky Allocation  = {opt_alloc:.3f}, Opt Val = {val:.3f}")
        print("Optimal Weights below:")
        for wts in q.weights:
            print(wts.weights)
