import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from numpy import array
from numpy import random, ones, zeros
from numpy.matlib import repmat
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

# Setting the random seed for reproducibility
random.seed(42)

# Global parameters
W_theorical = array([ -.3, .5 ])
noise_stdev = .25
n_samples = 25
x_dom = [0, 1]

# Some matplotlib general customization
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'


def h_theorical(X):
    return X @ W_theorical


def generate_noise(n):
    return random.normal(0, noise_stdev, n)


def generate_predictors(n):
    x = random.uniform(x_dom[0], x_dom[1], n)  # Between [0, 1]
    return array([ones(n), x]).T


def generate_samples(n):
    X = generate_predictors(n)
    T = h_theorical(X) + generate_noise(n)
    return (X, T)


def model_predict(W, X):
    # Linear regression model
    return X @ W


def loss(T_pred, T_theorical):
    # Sum of square errors computed for each column separately
    return np.sum((T_pred - T_theorical)**2, axis=0)


def loss_gradient(w, X, t):
    # Gradient of sum of square errors
    n = X.shape[0]  # Number of samples
    d = X.shape[1]  # Dimension of the predictors
    g = zeros(d)
    for i in range(n):
        x = X[i]
        g += float(x@w - t[i])*x
    return g


def calc_meshgrid(f, x_range, y_range):
    if x_range is None:
        if y_range is None:
            x_range = y_range = np.linspace(-1, 1, 25)
        else:
            x_range = y_range
    elif y_range is None:
        y_range = x_range

    X, Y = np.meshgrid(x_range, y_range)
    return (X, Y, f(X, Y))


def plot_contour(f, x_range=None, y_range=None, title=''):
    X, Y, Z = calc_meshgrid(f, x_range, y_range)

    plt.figure()
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title(title)


def plot_surface(f, x_range=None, y_range=None, title=''):
    X, Y, Z = calc_meshgrid(f, x_range, y_range)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=True)
    # Customize the z axis.
    # ax.set_zlim(Z_MIN, Z_MAX)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.01f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.title(title)


def loss_contour(w0, w1, X, T):
    W = array([w0.flatten(), w1.flatten()])
    T_pred = model_predict(W, X)
    if T.ndim == 1:
        T = T[None].T

    n = T_pred.shape[1]
    L = loss(T_pred, np.tile(T, (1, n)))
    L.shape = w0.shape
    return L


def gradient_descent(gradient_func, w0, learning_rate=1, epsilon=1e-3, max_ite=1000):
    ret_w = [ w0 ]
    ret_g = [ gradient_func(w0) ]
    for i in range(max_ite):
        # Computes one step of the gradient descent
        w = ret_w[-1]
        g = ret_g[-1]
        next_w = w - learning_rate * g
        next_g = gradient_func(next_w)
        ret_w.append(next_w)
        ret_g.append(next_g)

        # TODO: Both convergence test could be enhanced
        # 1st test: has w converged since last iteration
        # 2nd test: is gradient close to 0
        if    all(abs(w - next_w) <= epsilon) \
           or all(abs(next_g) <= epsilon):
            break

    ret_w = np.array(ret_w)
    ret_w.shape = (ret_w.shape[0], ret_w.shape[1])
    ret_g = np.array(ret_g)
    ret_g.shape = (ret_g.shape[0], ret_g.shape[1])
    return ret_w, ret_g


class StochasticGradient:
    def __init__(self, X, t):
        self._X = X
        self._t = t
        self._n = len(t)
        self._indexes = list(range(self._n))
        self._shuffleIndexes()

    def _shuffleIndexes(self):
        self._ite = 0
        random.shuffle(self._indexes)

    def __call__(self, w):
        i = self._indexes[self._ite]
        x = self._X[i, :]
        t = self._t[i]
        l = float(x@w - t)*x

        self._ite += 1
        if self._ite >= self._n:
            self._shuffleIndexes();

        return l


# # Sample X and T
X, T = generate_samples(n_samples)

# Batch Gradient descent
n = X.shape[0]
d = X.shape[1]
w0 = array([.9,.9])
w_batch, g_batch = gradient_descent(
    lambda w: loss_gradient(w, X, T),
    w0,
    learning_rate=.04)  # Found by trial and error

w_sto, g_sto = gradient_descent(
    StochasticGradient(X, T),
    w0,
    learning_rate=.12,
    max_ite=100000

)

plot_contour(lambda w0, w1: loss_contour(w0, w1, X, T), title='Loss Function')
plt.plot(w_batch[:,0], w_batch[:,1], 'g-', alpha=.75 )
plt.plot(w_sto[:,0], w_sto[:,1], 'r-', alpha=.75 )
plt.show()

plot_surface(lambda w0, w1: loss_contour(w0, w1, X, T), title='Loss Function')
plt.show()
