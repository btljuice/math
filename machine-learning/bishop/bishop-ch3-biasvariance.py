import numpy as np
from numpy import random, sin, pi, exp
import matplotlib.pyplot as plt

### Generate the data from the sinusoidal
n_datasets = 100
n_samples = 25
noise_variance = .25
x_dom = (0, 1)

def h(x): return sin(2*pi*x)
def gaussian_kernel(x, mu, s): return exp(-.5*((x - mu)/s)**2)  # basis function, a gaussian kernel

# Returns all basis functions comprise of n gaussians kernels between [0, 1] and
# + 1 bias parameter
# x is a vector
def basis(X):
    n_basis = 24
    kernel_variance = .2

    if np.isscalar(X):
        X = np.array([X])

    Phi = []
    for x in np.nditer(X):
        Phi.append(1) # bias parameter
        for mu in np.linspace(0, 1, n_basis):
            Phi.append(gaussian_kernel(x, mu, kernel_variance))

    Phi = np.array(Phi)
    Phi.shape = X.shape + (n_basis+1,)
    return Phi


def linreg_maxll(Phi, t, reg_param):
    n = Phi.shape[1]
    reg_identity = reg_param*np.asmatrix(np.eye(n))
    return (reg_identity + Phi.T * Phi).I * Phi.T * t


def linreg_predict(W, x, basis_func=basis):
    return np.asmatrix(basis_func(x))*W

def linreg_predict_mean(W, x, basis_func=basis):
    return np.mean(linreg_predict(W,x,basis_func), axis=1)

def linreg_bias2(datasets, reg_params):
    if np.isscalar(reg_params):
        reg_params = [ reg_params]
    bias_2 = []
    for r in reg_params:
        W = np.column_stack([ linreg_maxll(d['Phi'],d['t'], r) for d in datasets ])
        x = np.linspace(0,1,100)
        t_theorical = np.asmatrix(h(x)).T
        t_pred_mean = linreg_predict_mean(W,x)
        bias_2.append(np.mean(np.square(t_pred_mean - t_theorical)))
    return bias_2

def linreg_variance(datasets, reg_params):
    if np.isscalar(reg_params):
        reg_params = [reg_params]
    variance = []
    for r in reg_params:
        W = np.column_stack([ linreg_maxll(d['Phi'],d['t'], r) for d in datasets ])
        deviations = []
        for i in range(len(datasets)):
            w = W[:,i]
            x = datasets[i]['x']
            t_pred = linreg_predict(w, x)
            t_pred_mean = linreg_predict_mean(W, x)
            deviations.append(np.mean(np.square(t_pred - t_pred_mean)))
        variance.append(np.mean(deviations))
    return variance




### Generate samples h(x) + noise,
### where noise ~ Normal(0, noise_variance)
datasets = []
for i in range(n_datasets):
    noise = random.normal(0, noise_variance, n_samples)
    x = random.uniform(x_dom[0], x_dom[1], n_samples)  # Between [0, 1]
    t = h(x) + noise
    Phi = basis(x)
    datasets.append({
        'noise' : np.asmatrix(noise).T,
        'x' : np.asmatrix(x).T,
        't' : np.asmatrix(t).T,
        'Phi' : np.asmatrix(Phi)
    })

# Compute fig 3.5. plots
plt.figure(figsize=(10,10))
reg_coefs =  [exp(5), exp(3), exp(-5)]
plt_index = 0
x = np.linspace(0, 1, 100)
for c in reg_coefs:
    W = np.column_stack([ linreg_maxll(d['Phi'],d['t'], c) for d in datasets ])
    plt_index += 1
    ax = plt.subplot(320 + plt_index)
    ax.set_xlim([0,1])
    ax.set_ylim([-1.5,1.5])
    for i in range(10):
        w = W[:,i]
        plt.plot(x, linreg_predict(w, x), 'r', alpha=.3)
    plt_index += 1
    plt.subplot(320 + plt_index)
    ax.set_xlim([0,1])
    ax.set_ylim([-1.5,1.5])
    plt.plot(x, h(x), 'g' )
    plt.plot(x, linreg_predict_mean(W, x), 'b')
plt.show()

### Compute bias and variance
c = np.linspace(-10, 5, 100)
bias2 = np.array(linreg_bias2(datasets, np.exp(c) ))
variances = np.array(linreg_variance(datasets, np.exp(c) ))
plt.plot(c, bias2 , 'r')
plt.plot(c, variances, 'b')
plt.plot(c, bias2 + variances, 'g')
plt.gca().set_ylim([0,.15])
plt.show()

