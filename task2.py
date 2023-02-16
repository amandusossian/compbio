import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import odeint


def UpdateConcentrations(grid, tStep, Du, Dv, L, a, b):
    return grid + tStep*Diffusion(grid, Du, Dv, L, a, b)

def Diffusion(grid, Du, Dv, L, a, b):
    dU = np.zeros((L, L))
    dV = np.zeros((L, L))
    
    for i in range(L):
        for j in range(L):
             
            dU[i, j] = (a -(b+1)*grid[0, i, j] + grid[0, i, j]**2*grid[1, i, j] + Du*(grid[0, i, (j+1)%L] + 
                        grid[0, i, (j-1)%L] + grid[0, (i+1)%L, j] + grid[0, (i-1)%L, j] - 4*grid[0, i, j]))
            
            dV[i, j] = (b*grid[0, i, j]- grid[0, i, j]**2*grid[1, i, j] + Dv*(grid[1, i, (j+1)%L] + 
                        grid[1, i, (j-1)%L] + grid[1, (i+1)%L, j] + grid[1, (i-1)%L, j] - 4*grid[1, i, j]))
    
    return np.array([dU, dV])

def TaskC():
    a = 3
    b = 8
    Du = 1
    L = 128
    tStep = 0.01
    tIter = 10000
    Dvs = np.array([2.3, 3, 5, 9])
    grid = np.zeros((2, L, L))
    initialValueU = a
    initialValueV = b/a
    grid[0,:,:] =  (np.ones((L, L))+ np.random.normal(0, 0.1, (L, L)))*initialValueU
    grid[1,:,:] =  (np.ones((L, L))+ np.random.normal(0, 0.1, (L, L)))*initialValueV
    Dv = Dvs[1]
    temp_u = np.zeros((L, L))
    for i in range(tIter):
        
        grid = UpdateConcentrations(grid, tStep, Du, Dv, L, a, b)
        if (temp_u==grid[0,:,:]).all():
            print("u is constant at time step " + str(i))
            plt.pcolor(grid[0,:,:])
            plt.colorbar()
            plt.clim([0, 6])
            plt.title("Steady state reached at time t = " + str(0.01*(i+1)))
            plt.show()
            
            break
        temp_u = grid[0,:,:]
        if i%100 == 0:
            print(i)
        if (i+1)%1000 == 0:
            
            plt.pcolor(grid[0,:,:])
            plt.colorbar()
            plt.clim([0, 6])
            plt.title("u at time step " + str(i+1))
            plt.show()
            
    
TaskC()
def Test():
    a = np.array([[1,1,3],[4,5,6]])
    b = np.array([[1,2,3],[4,5,6]])
    print((a==b).all())

#Test()