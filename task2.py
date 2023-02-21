import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import odeint


def UpdateConcentrations(grid, tStep, Du, Dv, L, a, b):
    return grid + tStep*Diffusion(grid, Du, Dv, L, a, b)

def Diffusion(grid, Du, Dv, L, a, b):

    u = np.array(grid[0,:,:])

    uu = np.roll(u, shift =-1, axis  = 0)
    ud = np.roll(u, shift =1, axis  = 0)
    uf = np.roll(u, shift =1, axis  = 1)
    ub = np.roll(u, shift =-1, axis  = 1)

    v = np.array(grid[1,:,:])

    vu = np.roll(v, shift =-1, axis  = 0)
    vd = np.roll(v, shift =1, axis  = 0)
    vf = np.roll(v, shift =1, axis  = 1)
    vb = np.roll(v, shift =-1, axis  = 1)

    

    dU = np.ones((L,L))*a - (b+1)*u + np.power(u, 2)*v + Du*(uf + ub+ ud + uu - 4*u)
    dV = b*u- np.power(u, 2)*v + Dv* (vf + vb + vd + vu -4*v)

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
    Dv = Dvs[3]

    # Dvs[0] sets range to 3
    # Dvs[1] sets range to 4.5
    # Dvs[2] sets range to 7.5
    # Dvs[3] sets range to 12
    temp_u = np.zeros((L, L))
    i = 0
    while i< tIter:        
        grid = UpdateConcentrations(grid, tStep, Du, Dv, L, a, b)
        if (temp_u==grid[0,:,:]).all():
            print("u is constant at time step " + str(i))
            plt.pcolor(grid[0,:,:])
            plt.colorbar()
            plt.clim([2.9, 3.1])
            plt.title("Steady state reached at time t = " + str(0.01*(i+1)))
            plt.show()
            
            break
        temp_u = grid[0,:,:]
        if i%100 == 0:
            print(i)
        if (i+1)%5000 == 0:
            
            plt.pcolor(grid[0,:,:])
            plt.colorbar()
            plt.clim([0, 12])
            plt.title("u at time step " + str(i+1))
            plt.show()
        i+=1
            
TaskC()
def Test():
    L = 3
    a = 1
    b = 2
    Du = 2
    Dv = 2
    u = np.array([[1,2,3], [4,5,6], [7,8,9]])
    
    uu = np.roll(u, shift =-1, axis  = 0)
    ud = np.roll(u, shift =1, axis  = 0)
    uf = np.roll(u, shift =1, axis  = 1)
    ub = np.roll(u, shift =-1, axis  = 1)
    print(u)
    print(ub)
    print(uf)
    print(uu)
    print(ud)
    
    return
    v = np.array([[2,2,2], [3,3,3], [4,4,4]])
    vu = np.roll(u, shift =-1, axis  = 0)
    vd = np.roll(u, shift =1, axis  = 0)
    vf = np.roll(u, shift =1, axis  = 1)
    vb = np.roll(u, shift =-1, axis  = 1)

    dU = np.zeros((L,L))
    dV = np.zeros((L,L))
    print(a*np.ones((L,L)) + (b+1)*u )
    print(Du*(uf + ub+ ud + uu - 4*u))
    print(np.power(u, 2)*v)
    dU = a*np.ones((L,L)) - (b+1)*u + np.power(u, 2)*v + Du*(uf + ub+ ud + uu - 4*u)
    dV = b*u- np.power(u, 2)*v + Dv* (vf + vb + vd + vu -4*v)
    print(dU)
    print(dV)
    
#Test()