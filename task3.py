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
    t_iter = 1000
    tStepSize = 0.1
    k_vals = np.array([k_c*0.5, k_c*1.2, k_c*10])
    n_vals = np.array([20, 100, 300])

    fig, ax = plt.subplots(3, 3)

    for i, k in enumerate(k_vals):
        for j, n in enumerate(n_vals):
            thetas = np.zeros([n, t_iter])
            thetas[:, 0] = np.random.uniform(-np.pi/2, np.pi/2, size=n)
            rand_freq = cauchy.rvs(loc=0, scale=gamma, size=n)
            for t in range(1, t_iter):
                thetas[:, t] = UpdateThetas(thetas, rand_freq, tStepSize, gamma, k, n, t)

            r_param = OrderParameter(thetas, n, t_iter)
            ax[i, j].plot(np.linspace(0, t_iter, t_iter), r_param)
            ax[i, j].set_title("K = " + str(k) + ", N = " + str(n))
            if j == 0:
                ax[i, j].set_ylabel("r")
            if i == 2:
                ax[i, j].set_xlabel("Iteration")

    plt.show()

TaskB()
