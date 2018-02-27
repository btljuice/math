import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from scipy.stats import norm

nb_samples = 100
number_of_frames = nb_samples
mu = 10
std = 2.0
data = norm.rvs(mu, std, size=nb_samples)
min_val = np.min(data)
max_val = np.max(data)
x = np.linspace(min_val, max_val, 100)
leg = [ mpatches.Patch(color='red', label='theorical'),
        mpatches.Patch(color='green', label='likelihood') ]


def update_hist(num, data):
    ll_mu = np.mean(data[1:num])
    ll_std = np.std(data[1:num], ddof=1)
    plt.cla()  # other options clf(), close()
    axes = plt.gca()
    axes.set_xlim(min_val, max_val)
    axes.set_ylim(0,.25)
    plt.hist(data[1:num], density=True)
    plt.plot(x, norm.pdf(x, mu, std), 'r')
    plt.plot(x, norm.pdf(x, ll_mu, ll_std), 'g')
    plt.legend(handles=leg)


fig = plt.figure()
hist = plt.hist(data[0])

anim = animation.FuncAnimation(fig, update_hist, number_of_frames, fargs=(data, ),
                                    interval=100, repeat=False)
anim.save('gaussian-convergence.gif', writer='imagemagick',fps=10)
plt.show()
