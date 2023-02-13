from dataclasses import dataclass
from rl.markov_decision_process import MarkovDecisionProcess
from rl.markov_process import Terminal, NonTerminal, State
from rl.distribution import Distribution, Gaussian, SampledDistribution
from rl.function_approx import LinearFunctionApprox
from rl.approximate_dynamic_programming import ValueFunctionApprox, back_opt_vf_and_policy
from rl.policy import DeterministicPolicy

from typing import Iterable, TypeVar, Tuple, Callable, Iterator, Sequence

A = TypeVar("A")
S = TypeVar("S")

@dataclass
class OptimalOptionExecution(MarkovDecisionProcess[float, int]):
    new_price_dist: Sequence[Callable[[float], Distribution]]
    strike_price: float
    call_option: bool
    t: int

    def actions(self, state: NonTerminal[S]) -> Iterable[A]:
        acts = [0, 1]
        return acts

    def step(self, state: NonTerminal[S], action: A) -> Distribution[Tuple[State[S], float]]:
        cur_price = state.state
        def calc_payoff(cur_price=cur_price) -> Tuple[State[S], float]:
            new_price = self.new_price_dist[self.t](cur_price).sample()
            if action==1:
                if self.call_option:
                    payoff = max(cur_price - self.strike_price, 0)
                else:
                    payoff = max(self.strike_price - new_price, 0)

                next_state: State[float] = Terminal(new_price)
            else:
                payoff = 0
                next_state: State[float] = NonTerminal(new_price)
            
            return (next_state, payoff)

        return SampledDistribution(sampler=calc_payoff, expectation_samples=10)
    
def get_payoff_distribution(t: int, new_price_dists) -> \
            SampledDistribution[NonTerminal[float]]:
    def get_payoff() -> NonTerminal[float]:
        price = 100
        for i in range(t):
            price = new_price_dists[i](price).sample()
        return NonTerminal(price)

    return SampledDistribution(get_payoff)
  
new_price_dists = [lambda mu: Gaussian(μ=mu, σ=1.0) for _ in range(5)]
ooe_mdps = []
payoff_dists = []
for i in range(5):
    ooe_mdps.append(OptimalOptionExecution(new_price_dist=new_price_dists, strike_price=100, call_option=True, t=i))
    payoff_dists.append(get_payoff_distribution(i, new_price_dists))

feature_functions = [lambda state: state.state]
linear_function_approx = LinearFunctionApprox.create(feature_functions=feature_functions)

mdp_f0_mu_triples: Sequence[Tuple[
    MarkovDecisionProcess[float, bool],
    ValueFunctionApprox[float],
    SampledDistribution[NonTerminal[float]]
]] = [(ooe_mdps[i], linear_function_approx, payoff_dists[i]) for i in range(5)]

num_state_samples: int = 5000
error_tolerance: float = 1e-3

vf_iter = back_opt_vf_and_policy(
            mdp_f0_mu_triples=mdp_f0_mu_triples,
            γ=1.0,
            num_state_samples=num_state_samples,
            error_tolerance=error_tolerance
        )

price_to_investigate = 100.5

for time_step, (option_value, policy) in enumerate(vf_iter):
    print(f"At Time {time_step:d} and price {price_to_investigate}:")
    exercise: int = policy.action_for(price_to_investigate)
    val: float = option_value(NonTerminal(price_to_investigate))
    print(f"Option Value = {val:.3f} and Exercise = {exercise}")