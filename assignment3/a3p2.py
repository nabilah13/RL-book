import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Set the input parameters
r = 0.03
u = 0.05
sigmasq = 0.02
alphas = np.arange(0.3, 1.5, 0.1)

pi = (u-r)*(1-alphas*(1+r)) / (alphas*((u-r)**2)+alphas*sigmasq)
pi_dollars = 1e6 * pi

fig, ax = plt.subplots()
ax.scatter(alphas, pi_dollars,c='green')
ax.set_title(f"Optimal Investment in Risky Asset: r={r}, u={u}, sigmasq={sigmasq}")
ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('${x:,.0f}'))
ax.set_xlabel("Alpha Value")
ax.set_ylabel("Dollars Invested in Risky Asset")
fig.set_figwidth(12)
fig.set_figheight(8)
plt.savefig("assignment3/optimal_dollars.png")
plt.clf()

