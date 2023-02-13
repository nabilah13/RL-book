from rl.chapter9.order_book import OrderBook, DollarsAndShares, PriceSizePairs
from rl.markov_process import S, MarkovProcess, NonTerminal
from rl.distribution import Distribution, SampledDistribution, Poisson, Gaussian, Constant
from dataclasses import dataclass
from operator import itemgetter
from copy import deepcopy


@dataclass
class OrderBookDynamics(MarkovProcess[OrderBook]):
    num_lim_buys_dist: Distribution[int]
    num_lim_sells_dist: Distribution[int]

    size_lim_buys_dist: Distribution[int]
    size_lim_sells_dist: Distribution[int]

    num_mark_buys_dist: Distribution[int]
    num_mark_sells_dist: Distribution[int]

    size_mark_buys_dist: Distribution[int]
    size_mark_sells_dist: Distribution[int]

    price_lim_buys_dist: Distribution[float]
    price_lim_sells_dist: Distribution[float]

    def transition(self, state: NonTerminal[OrderBook]) -> Distribution[OrderBook]:
        def update_order_book():
            new_book = deepcopy(state.state)
            
            for mark_buy in range(self.num_mark_buys_dist.sample()):
                size_mark_buy = self.size_mark_buys_dist.sample()
                new_book = new_book.buy_market_order(size_mark_buy)[1]

            for mark_sell in range(self.num_mark_sells_dist.sample()):
                size_mark_sell = self.size_mark_sells_dist.sample()
                new_book = new_book.sell_market_order(size_mark_sell)[1]    
            
            # Get price to base the limit buys and sells off of
            if len(new_book.ascending_asks) == 0:
                cur_price = new_book.descending_bids[0].dollars
            elif len(new_book.descending_bids) == 0:
                cur_price = new_book.ascending_asks[0].dollars
            else:
                cur_price = new_book.mid_price()

            cur_price = round(cur_price, 2)

            for lim_buy in range(self.num_lim_buys_dist.sample()):
                lim_buy_price = round(max(cur_price + self.price_lim_buys_dist.sample(), 0),2)
                size_lim_buy = self.size_lim_buys_dist.sample()
                new_book = new_book.buy_limit_order(lim_buy_price, size_lim_buy)[1]

            for lim_sell in range(self.num_lim_sells_dist.sample()):
                lim_sell_price = round(max(cur_price + self.price_lim_sells_dist.sample(), 0),2)
                size_lim_sell = self.size_lim_sells_dist.sample()
                new_book = new_book.sell_limit_order(lim_sell_price, size_lim_sell)[1]

            return NonTerminal(new_book)
         
        return SampledDistribution(update_order_book)


# Set up initial OrderBook
bids = [(round(Gaussian(99, 1).sample(),2), Poisson(25).sample()) for _ in range(10)]
asks = [(round(Gaussian(101, 1).sample(),2), Poisson(25).sample()) for _ in range(10)]
bids = sorted(bids,key=itemgetter(0), reverse=True)
asks = sorted(asks, key=itemgetter(0), reverse=False)
asks = [DollarsAndShares(x[0], x[1]) for x in asks]
bids =  [DollarsAndShares(x[0], x[1]) for x in bids]

initial_order_book: OrderBook = OrderBook(descending_bids=bids, ascending_asks=asks)

# Set up buy/sell distributions
num_lim_buys_dist: Distribution[int] = Poisson(3)
num_lim_sells_dist: Distribution[int] = Poisson(6)
size_lim_buys_dist: Distribution[int] = Poisson(5)
size_lim_sells_dist: Distribution[int] = Poisson(5)
num_mark_buys_dist: Distribution[int] = Poisson(3)
num_mark_sells_dist: Distribution[int] = Poisson(3)
size_mark_buys_dist: Distribution[int] = Poisson(5)
size_mark_sells_dist: Distribution[int] = Poisson(5)
price_lim_buys_dist: Distribution[float] = Gaussian(-5, 1)
price_lim_sells_dist: Distribution[float] = Gaussian(-5, 1)


order_book_dynamics = OrderBookDynamics(num_lim_buys_dist, num_lim_sells_dist, 
                                        size_lim_buys_dist, size_lim_sells_dist, 
                                        num_mark_buys_dist, num_mark_sells_dist,
                                        size_mark_buys_dist, size_mark_sells_dist, 
                                        price_lim_buys_dist, price_lim_sells_dist)

traces = order_book_dynamics.traces(Constant(NonTerminal(initial_order_book)))

simulation = next(traces)
for j in range(10):
    cur_order_book = next(simulation)
    print("Time Step: {}".format(j))
    print("OrderBook")
    cur_order_book.state.pretty_print_order_book()

cur_order_book.state.display_order_book()


