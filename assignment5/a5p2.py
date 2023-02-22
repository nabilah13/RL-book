import numpy as np
import pandas as pd
from dataclasses import dataclass
from matplotlib import pyplot as plt


@dataclass
class State:
    time: float
    midpoint: float
    bid: float
    ask: float
    profit: float
    inventory_on_hand: int
    num_hits: int
    num_lifts: int


class OptimalMarketMaker():
    def __init__(self, S_0: float, T: float, dt: float, 
                 gamma: float, sigma: float, I_0: int, k: float, 
                 c: float, naive: bool = False,
                 avg_bid_ask_spread: float = 0):
        self.S_0 = S_0
        self.T = T
        self.dt = dt
        self.gamma = gamma
        self.sigma = sigma
        self.I_0 = I_0
        self.k = k
        self.c = c
        self.naive = naive
        self.avg_bid_ask_spread = avg_bid_ask_spread
        # self.midpoint = S_0
        # self.profit = 0

    def get_bid(self, t: float, inventory: int, midpoint: float):
        if self.naive:
            return midpoint - self.avg_bid_ask_spread / 2
        bid_helper = (2 * inventory + 1) * self.gamma * \
            (self.sigma)**2 * (self.T - t) / 2
        return midpoint - (bid_helper + np.log(1 + self.gamma / self.k) / self.gamma)

    def get_ask(self, t: float, inventory: int, midpoint: float):
        if self.naive:
            return midpoint + self.avg_bid_ask_spread / 2
        ask_helper = (1 - 2 * inventory) * self.gamma * \
            (self.sigma)**2 * (self.T - t) / 2
        return midpoint + (ask_helper + np.log(1 + self.gamma / self.k) / self.gamma)

    def get_state(self, t: float, inventory: int, midpoint: float, profit: float, num_hits: int, num_lifts: int):
        return State(t, midpoint,
                     self.get_bid(t, inventory, midpoint),
                     self.get_ask(t, inventory, midpoint),
                     profit, inventory, num_hits, num_lifts)

    def run_trace(self):
        t = 0
        inventory = 0
        states = []
        midpoint = self.S_0
        profit = 0
        num_hits = 0
        num_lifts = 0
        for t in np.arange(0, self.T, self.dt):
            states.append(self.get_state(
                t, inventory, midpoint, profit, num_hits, num_lifts))
            bid = states[-1].bid
            ask = states[-1].ask
            # hit prob
            hit_prob = (self.c * np.exp(-1.0 * self.k *
                        (ask-midpoint)) * self.dt)
            # lifted prob
            lifted_prob = (
                self.c * np.exp(-1.0 * self.k * (midpoint-bid)) * self.dt)
            # if hit
            if np.random.random() < hit_prob:
                inventory += 1
                num_hits += 1
                profit -= bid
            # if lifted
            if np.random.random() < lifted_prob:
                inventory -= 1
                num_lifts += 1
                profit += ask
            midpoint += (1 if np.random.random() > 0.5 else -1) * \
                self.sigma * np.sqrt(self.dt)
            t += self.dt
            inventory += 1
        # at end of trace sell all inventory. or buy all inventory if negative
        profit += inventory * midpoint
        states.append(self.get_state(
            t, inventory, midpoint, profit, num_hits, num_lifts))
        return states


if __name__ == "__main__":
    params = {
        'S_0': 100,
        'T': 1,
        'dt': 0.005,
        'gamma': 0.1,
        'sigma': 2,
        'I_0': 0,
        'k': 1.5,
        'c': 140
    }
    om = OptimalMarketMaker(**params)
    optimal_traces = []
    avg_bid_ask_spread = []
    for i in range(100):
        optimal_traces.append(om.run_trace())
        avg_bid_ask_spread.append(
            np.mean([t.ask - t.bid for t in optimal_traces[-1]]))

    bid_ask_mean = np.mean(avg_bid_ask_spread)
    print(
        f"Using average bid-ask spread of {bid_ask_mean} for naive market maker.")
    params['naive'] = True
    params['avg_bid_ask_spread'] = bid_ask_mean
    om = OptimalMarketMaker(**params)
    traces_naive = []
    for i in range(100):
        traces_naive.append(om.run_trace())

    # profit comparison
    naive_profit = [t[-1].profit for t in traces_naive]
    optimal_profit = [t[-1].profit for t in optimal_traces]
    print(f"Naive market maker average profit: {np.mean(naive_profit)}")
    print(f"Optimal market maker average profit: {np.mean(optimal_profit)}")

    # inventory comparison
    naive_inventory = [t[-1].inventory_on_hand for t in traces_naive]
    optimal_inventory = [t[-1].inventory_on_hand for t in optimal_traces]
    print(f"Naive market maker average inventory: {np.mean(naive_inventory)}")
    print(
        f"Optimal market maker average inventory: {np.mean(optimal_inventory)}")

    # OB Mid Price comparison
    naive_mp = [t[-1].midpoint for t in traces_naive]
    optimal_mp = [t[-1].midpoint for t in optimal_traces]
    print(f"Naive market maker average OB Mid Price: {np.mean(naive_mp)}")
    print(f"Optimal market maker average OB Mid Price: {np.mean(optimal_mp)}")

    # bid comparison
    naive_bidprice = [t[-1].bid for t in traces_naive]
    optimal_bidprice = [t[-1].bid for t in optimal_traces]
    print(f"Naive market maker average bid price: {np.mean(naive_bidprice)}")
    print(
        f"Optimal market maker average bid price: {np.mean(optimal_bidprice)}")

    # num lifts comparison
    naive_lifts = [t[-1].num_lifts for t in traces_naive]
    optimal_lifts = [t[-1].num_lifts for t in optimal_traces]
    print(f"Naive market maker average num lifts: {np.mean(naive_lifts)}")
    print(f"Optimal market maker average num lifts: {np.mean(optimal_lifts)}")

    # num hits comparison
    naive_hits = [t[-1].num_hits for t in traces_naive]
    optimal_hits = [t[-1].num_hits for t in optimal_traces]
    print(f"Naive market maker average num hits: {np.mean(naive_hits)}")
    print(f"Optimal market maker average num hits: {np.mean(optimal_hits)}")

    naive_metrics = [naive_profit, naive_inventory,
                     naive_mp, naive_bidprice, naive_lifts, naive_hits]
    optimal_metrics = [optimal_profit, optimal_inventory,
                       optimal_mp, optimal_bidprice, optimal_lifts, optimal_hits]
    metrics = ['Profit', 'Inventory', 'Midpoint', 'Bid Price', 'Lifts', 'Hits']
    for i, metric in enumerate(metrics):
        plt.figure()
        plt.hist(naive_metrics[i], alpha=0.5, label='Naive')
        plt.hist(optimal_metrics[i], alpha=0.5, label='Optimal')
        plt.legend(loc='upper right')
        plt.title(metric)
