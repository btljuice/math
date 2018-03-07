from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np
from numpy.random import uniform, normal
from numpy import matrix
from itertools import chain

# For reproducibility
np.random.seed(42)


# Plot the data
def plot3dFunction(ax, f, x0, x1, y0, y1, col, size=10):
    x = np.linspace(x0, x1, size)
    y = np.linspace(y0, y1, size)
    x, y = np.meshgrid(x, y)
    z = f(x, y)
    ax.plot_surface(x, y, z, alpha=.2, color=col)


def plotLinearRegression(ax, w, x1, x2, t):
    plot3dFunction(ax, plane(w_theo), *plot_xlim, *plot_ylim, 'red')
    plot3dFunction(ax, plane(w), *plot_xlim, *plot_ylim, 'green')
    ax.scatter(x1, x2, t, c='r', alpha=.2)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('t')


def columnsRange(x):
    return [(x[:, i].min(), x[:, i].max()) for i in range(x.shape[1])]


# Linear Regression with maximum likelihood estimators
def linreg_maxll(x, t):
    return (x.T * x).I * x.T * t


def plane(w):
    return lambda x, y: float(w[0]) + float(w[1])*x + float(w[2])*y


# Animation parameters
nb_frames = 10
nb_samples = 10
animation_fps = 1

# Updates animation
def animation_update(i, ax, x, t):
    m = int(i/nb_frames * nb_samples)
    if (m < 2):
        return
    plt.cla()  # other options clf(), close()
    ax.set_xlim(*plot_xlim)
    ax.set_ylim(*plot_ylim)
    ax.set_zlim(*plot_zlim)
    plotLinearRegression(ax, linreg_maxll(x[:m,],t[:m]), x[:m,1], x[:m,2], t[:m])


# Create samples following a linear plane with gaussian noise
w_theo = matrix(uniform(-10, 10, 3)).T  # w0 + w1*x1 + w2*x2
beta = uniform(20, 50)
x = matrix([np.ones(nb_samples),
            uniform(-20, 20, nb_samples),      # x1 on the x axis
            uniform(-20, 20, nb_samples) ]).T  # x2 on the y axis
noise = matrix(normal(0, beta, nb_samples)).T
t = x * w_theo + noise


# Plot plane and samples
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plot_xlim = (x[:,1].min(), x[:,1].max())
plot_ylim = (x[:,2].min(), x[:,2].max())
plot_zlim = (t.min(), t.max())

anim = animation.FuncAnimation(fig, animation_update, nb_frames,
                               fargs=(ax, x, t),
                               interval=1000/animation_fps, repeat=False)
anim.save('linreg-maxll.mp4', writer='ffmpeg', fps=animation_fps)
plt.show()
