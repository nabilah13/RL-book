# adding src to the system path
import sys
sys.path.insert(0, '/Users/nabilahmed/RL-book/')

from typing import Mapping, Dict, Tuple
from rl.distribution import Categorical
from rl.markov_process import FiniteMarkovRewardProcess
from typing import Mapping, Dict, Tuple, Sequence, Iterator, TypeVar
from rl.chapter10.prediction_utils import (mc_finite_prediction_learning_rate, 
                                           td_finite_prediction_learning_rate,
                                           td_lambda_finite_prediction_learning_rate)
from rl.approximate_dynamic_programming import ValueFunctionApprox
from rl.markov_process import NonTerminal 
import itertools
from math import sqrt
from rl.returns import returns
from rl.distribution import Choose

S = TypeVar('S')

class RandomWalkMRP2D(FiniteMarkovRewardProcess[int]):
    '''
    For the 1D case:
    This MRP's states are {0, 1, 2,...,self.barrier}
    with 0 and self.barrier as the terminal states.
    At each time step, we go from state i to state
    i+1 with probability self.p or to state i-1 with
    probability 1-self.p, for all 0 < i < self.barrier.
    The reward is 0 if we transition to a non-terminal
    state or to terminal state 0, and the reward is 1
    if we transition to terminal state self.barrier

    For the 2D case:
    This MRP's states are {(i, j) | 0 <= i <= self.barrier_x, 0 <= j <= self.barrier_y}
    with (i, 0), (0, j), (i, self.barrier_y), (self.barrier_x, j) as the terminal states.
    At each time step, we go from state (i, j) to state
    (i+1, j+1) with probability self.p_x * self.p_y,  or to state (i-1, j-1) with
    probability (1-self.p_x) * (1-self.p_y), or to state (i+1, j-1) with
    probability self.p_x * (1-self.p_y), or to state (i-1, j+1) with
    probability (1-self.p_x) * self.p_y, for all 0 < i < self.barrier_x, 0 < j < self.barrier_y.
    The reward is 0 if we transition to a non-terminal
    state or to terminal state (0, j) or (i, 0), and the reward is 1
    if we transition to terminal state (self.barrier_x, j) or (i, self.barrier_y)
    '''
    barrier: int
    p_x: float
    p_y: float

    def __init__(
        self,
        barrier_x: int,
        barrier_y: int,
        p_inc: float,
        p_hori: float
    ):
        self.barrier_x = barrier_x
        self.barrier_y = barrier_y
        self.p_inc = p_inc
        self.p_hori = p_hori
        super().__init__(self.get_transition_map())

    def get_transition_map(self) -> \
            Mapping[Tuple[int, int], Categorical[Tuple[Tuple[int, int], float]]]:
        d: Dict[Tuple[int, int], Categorical[Tuple[Tuple[int, int], float]]] = {
            (i, j): Categorical({
                ((i + 1, j), 0. if (i < self.barrier_x - 1) else 1.): self.p_inc * self.p_hori,
                ((i - 1, j), 0.): (1. - self.p_inc) * self.p_hori,
                ((i, j + 1), 0. if (j < self.barrier_y - 1) else 1.): self.p_inc * (1. - self.p_hori),
                ((i, j - 1), 0.): (1. - self.p_inc) * (1. - self.p_hori)
            }) for i in range(1, self.barrier_x) for j in range(1, self.barrier_y)
        }
        return d

def compare_td_and_td_lambda_and_mc(
    fmrp: FiniteMarkovRewardProcess[S],
    gamma: float,
    mc_episode_length_tol: float,
    num_episodes: int,
    learning_rates: Sequence[Tuple[float, float, float]],
    initial_vf_dict: Mapping[NonTerminal[S], float],
    plot_batch: int,
    plot_start: int
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
    
    for k, (init_lr, half_life, exponent) in enumerate(learning_rates):
        td_lambda_funcs_it: Iterator[ValueFunctionApprox[S]] = \
            td_lambda_finite_prediction_learning_rate(
                fmrp=fmrp,
                gamma=gamma,
                lambd=0.5,
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
        label = f"TD Lambda InitRate={init_lr:.3f},HalfLife" + \
            f"={half_life:.0f},Exp={exponent:.1f}"
        plt.plot(
            range(len(td_lambda_plot)),
            td_lambda_plot,
            color=colors[k],
            linestyle=':',
            label=label
        )

    plt.xlabel("Episode Batches", fontsize=20)
    plt.ylabel("Value Function RMSE", fontsize=20)
    plt.title(
        "RMSE of MC, TD, and TD(0.5) as function of episode batches",
        fontsize=25
    )
    plt.grid(True)
    plt.legend(fontsize=10)
    plt.savefig("assignment6/convergence_rates_randomwalk_rmse.png")

if __name__ == '__main__':
    barrier_x: int = 10
    barrier_y: int = 10
    p_inc: float = 0.5
    p_hori: float = 0.5
    random_walk: RandomWalkMRP2D = RandomWalkMRP2D(
        barrier_x=barrier_x,
        barrier_y=barrier_y,
        p_inc=p_inc,
        p_hori=p_hori
    )
    compare_td_and_td_lambda_and_mc(
        fmrp=random_walk,
        gamma=1.0,
        mc_episode_length_tol=1e-6,
        num_episodes=700,
        learning_rates=[(0.01, 1e8, 0.5), (0.05, 1e8, 0.5)],
        initial_vf_dict={s: 0.5 for s in random_walk.non_terminal_states},
        plot_batch=7,
        plot_start=0
    )