from rl.td import q_learning_external_transitions, least_squares_policy_iteration
from rl.function_approx import LinearFunctionApprox, DNNApprox, DNNSpec, AdamGradient
from rl.distribution import Choose, SampledDistribution, Distribution
from rl.markov_process import State, NonTerminal, Terminal
from rl.markov_decision_process import MarkovDecisionProcess, TransitionStep
from rl.iterate import last
from rl.policy import  DeterministicPolicy, Always
from typing import Callable, Iterable, Sequence, Tuple, Iterator, List
import numpy as np
import itertools


def get_mdp(expiry, num_steps, payoff, rate, vol) -> MarkovDecisionProcess[float, bool]:
    dt: float = expiry / num_steps
    exer_payoff: Callable[[float], float] = payoff
    r: float = rate
    s: float = vol

    class OptExerciseBIMDP(MarkovDecisionProcess[float, bool]):
        def step(
                self,
                price: NonTerminal[float],
                exer: bool
        ) -> SampledDistribution[Tuple[State[float], float]]:

            def sr_sampler_func(
                    price=price,
                    exer=exer
            ) -> Tuple[State[float], float]:
                if exer:
                    return Terminal(0.), exer_payoff(price.state)
                else:
                    next_price: float = np.exp(np.random.normal(
                        np.log(price.state) + (r - s * s / 2) * dt,
                        s * np.sqrt(dt)
                    ))
                    return NonTerminal(next_price), 0.

            return SampledDistribution(
                sampler=sr_sampler_func,
                expectation_samples=200
            )

        def actions(self, price: NonTerminal[float]) -> Sequence[bool]:
            return [True, False]
    return OptExerciseBIMDP()


def lspi_features(
        T: int,
        K: float  # strike price
) -> Sequence[Callable[[Tuple[NonTerminal[float], int]], float]]:
    ret: List[Callable[[Tuple[NonTerminal[int], int]], float]] = [lambda x: 1.0]

    def first(x: Tuple[NonTerminal[float], int]) -> float:
        s, t = x
        M = s.state / K
        return np.exp(-(M / 2.0))

    ret.append(first)

    def second(x: Tuple[NonTerminal[float], int]) -> float:
        s, t = x
        M = s.state / K
        return first(x) * (1.0 - M)

    ret.append(second)

    def third(x: Tuple[NonTerminal[float], int]) -> float:
        s, t = x
        M = s.state / K
        return second(x) * (1.0 - M)

    ret.append(third)

    def fourth(x: Tuple[NonTerminal[float], int]) -> float:
        s, t = x
        M = s.state / K
        return first(x) * (1.0 - 2.0 * M + M * M / 2.0)

    ret.append(fourth)

    def fifth(x: Tuple[NonTerminal[float], int]) -> float:
        s, t = x
        M = s.state / K
        return np.sin(np.pi * (T - t) / (2 * T))

    ret.append(fifth)

    def sixth(x: Tuple[NonTerminal[float], int]) -> float:
        s, t = x
        return np.log(T - t)

    ret.append(sixth)

    def seventh(x: Tuple[NonTerminal[float], int]) -> float:
        s, t = x
        return (t / T) * (t / T)

    ret.append(seventh)
    return ret


def lspi_transitions(mdp: MarkovDecisionProcess) -> Iterator[TransitionStep[float, bool]]:
    def sample_price() -> NonTerminal[float]:
        return NonTerminal(np.random.uniform(0, 200))

    states_distribution: Distribution[NonTerminal[float]] = SampledDistribution(sample_price)
    while True:
        state: NonTerminal[float] = states_distribution.sample()
        action: bool = Choose([True, False]).sample()
        next_state, reward = mdp.step(state, action).sample()
        transition: TransitionStep[float, bool] = TransitionStep(
            state=state,
            action=action,
            next_state=next_state,
            reward=reward
        )
        yield transition


def lspi_vf_and_policy(mdp: MarkovDecisionProcess, T: int, K: float) -> Tuple[
    Callable[[NonTerminal[float]], float], DeterministicPolicy[int, int]]:
    transitions: Iterable[TransitionStep[float, bool]] = itertools.islice(
        lspi_transitions(mdp),
        20000
    )
    qvf_iter: Iterator[LinearFunctionApprox[Tuple[
        NonTerminal[float], bool]]] = least_squares_policy_iteration(
        transitions=transitions,
        actions=mdp.actions,
        feature_functions=lspi_features(T, K),
        initial_target_policy=Always(True),
        γ=1.0,
        ε=1e-5
    )
    qvf: LinearFunctionApprox[Tuple[NonTerminal[float], bool]] = \
        last(
            itertools.islice(
                qvf_iter,
                20
            )
        )

    def choose_action(state: NonTerminal[float]) -> bool:
        return qvf((state, False)) < mdp.step(state, True).expectation(
            lambda next_state_reward: next_state_reward[1]
        )

    policy = DeterministicPolicy(choose_action)

    def vf(state: NonTerminal[float]) -> float:
        if choose_action(state):
            return mdp.step(state, True).expectation(lambda next_state_reward: next_state_reward[1])
        else:
            return qvf((state, False))

    return vf, policy


if __name__ == '__main__':
    spot_price_val: float = 100.0
    strike: float = 100.0
    is_call: bool = False
    expiry_val: float = 1.0
    rate_val: float = 0.05
    vol_val: float = 0.25
    num_steps_val: int = 300

    if is_call:
        opt_payoff = lambda x: max(x - strike, 0)
    else:
        opt_payoff = lambda x: max(strike - x, 0)

    mdp_options = get_mdp(expiry_val, num_steps_val, opt_payoff, rate_val, vol_val)

    # simulate actions of mdp
    features = lspi_features(num_steps_val, strike)
    vf_final, pol = lspi_vf_and_policy(mdp_options, num_steps_val, strike)

    print("Value function at spot price with lpsi: ", vf_final(NonTerminal(spot_price_val)))

    # call q learning with external transitions for the above mdp
    reg: float = 1e-2
    neurons: Sequence[int] = [6]
    num_laguerre: int = 2
    ident: np.ndarray = np.eye(num_laguerre)
    dnn_spec = DNNSpec(
        neurons=neurons,
        bias=True,
        hidden_activation=lambda x: np.log(1 + np.exp(-x)),
        hidden_activation_deriv=lambda y: np.exp(-y) - 1,
        output_activation=lambda x: x,
        output_activation_deriv=lambda y: np.ones_like(y)
    )

    approx_0 = DNNApprox.create(
        feature_functions=features,
        dnn_spec=dnn_spec,
        adam_gradient=AdamGradient(
            learning_rate=0.1,
            decay1=0.9,
            decay2=0.999
        ),
        regularization_coeff=reg
    )
    it = q_learning_external_transitions(lspi_transitions(mdp_options), mdp_options.actions, approx_0, 1.0)

    # print the value function at the spot price (runs very long but works)
    print("Value function at spot price with q learning: ", last(it)(NonTerminal(spot_price_val)))