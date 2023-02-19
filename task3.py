import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.stats import cauchy


def UpdateThetas(theta, rand_freq, tStepSize, gamma, K, N, t):
    theta_change = np.zeros(N)
    for i, theta_i in enumerate(theta[:, t-1]):
        theta_change[i] = rand_freq[i] + K/N * np.sum(np.sin(theta_i-theta[:, t-1]))  # Kanske behöver tänka på vinkelsubtraktion
    return theta[:, t-1] + tStepSize * theta_change


def TaskB():
    gamma = 0.1
    k_c = 2*gamma
    t_iter = 200
    tStepSize = 0.1
    k_vals = np.array([k_c*0.5, k_c*1.01, k_c*2])
    n_vals = np.array([20, 100, 300])

    k = k_vals[0]
    n = n_vals[0]

    thetas = np.zeros([n, t_iter])
    thetas[:, 0] = np.random.uniform(-np.pi/2, np.pi/2, size=n)

    for t in range(1, t_iter):
        rand_freq = cauchy.rvs(loc=0, scale=gamma, size=n)
        thetas[:, t] = UpdateThetas(thetas, rand_freq, tStepSize, gamma, k, n, t)

    print("Done")


TaskB()


def Test():
    a = np.array([[1, 1, 3], [4, 5, 6]])
    b = np.array([[1, 2, 3], [4, 5, 6]])
    print((a == b).all())

# Test()