import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.stats import cauchy


def UpdateThetas(theta, rand_freq, tStepSize, gamma, K, N, t):
    theta_change = np.zeros(N)
    for i, theta_i in enumerate(theta[:, t-1]):
        theta_change[i] = rand_freq[i] + K/N * np.sum(np.sin(np.angle(np.exp(1j*theta_i)-np.exp(1j*theta[:, t-1]))))  # Kanske behöver tänka på vinkelsubtraktion
    return theta[:, t-1] + tStepSize * theta_change

def OrderParameter(thetas, N, t_iter):
    r_param = np.zeros([t_iter])
    thetas = thetas % 2*np.pi - np.pi
    for t in range(t_iter):
        r_param[t] = np.absolute(1/N*np.sum(np.exp(1j*thetas[:, t])))

    return r_param

def TaskB():
    gamma = 0.1
    k_c = 2*gamma
    t_iter = 200
    tStepSize = 0.1
    k_vals = np.array([k_c*0.5, k_c*1.01, k_c*4])
    n_vals = np.array([20, 100, 300])

    k = k_vals[2]
    n = n_vals[2]

    thetas = np.zeros([n, t_iter])
    thetas[:, 0] = np.random.uniform(-np.pi/2, np.pi/2, size=n)

    for t in range(1, t_iter):
        rand_freq = cauchy.rvs(loc=0, scale=gamma, size=n)
        thetas[:, t] = UpdateThetas(thetas, rand_freq, tStepSize, gamma, k, n, t)

    r_param = OrderParameter(thetas, n, t_iter)
    plt.plot(np.linspace(0, t_iter, t_iter), r_param)
    plt.show()


TaskB()


def Test():
    a = np.array([[1, 1, 3], [4, 5, 6]])
    b = np.array([[1, 2, 3], [4, 5, 6]])
    print((a == b).all())

# Test()