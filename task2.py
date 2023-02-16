import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import odeint


def taskb():
    a = 3
    b = 8
    Du = 1

def UpdateConcentrations():
    pass

def taskc():
    a = 3
    b = 8
    Du = 1
    L = 128
    t_step = 0.01
    t_iter = 5000
    Dvs = np.array([2.3, 3, 5, 9])
    grid = np.zeros((L, L, 2))
    initial_value_u = a
    initial_value_v = b/a
    grid[:,:,0] = grid[:,:,0] + (np.ones((L, L))+ np.random.normal(0, 0.1, (L, L)))*initial_value_u
    grid[:,:,1] = grid[:,:,1] + (np.ones((L, L))+ np.random.normal(0, 0.1, (L, L)))*initial_value_v
    Dv = Dvs[0]

    for i in range(t_iter):
        grid = UpdateConcentrations(grid, Du, L, t_step)
def test():
    L = 2
    for i in range(4):
        print(np.random.normal(0, 0.1, (L, L)))
test()