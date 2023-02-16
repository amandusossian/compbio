import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import odeint

def taska():
    r, q, u = sp.symbols('r q u')
    f = r+ r*u - r/q*u - r/q*u**2 - 1
    solu = sp.solve(f, u)
    sp.pprint(solu)
#taska()


def initializeU(eps0, u0, L, t):
    u = np.zeros((L, t), dtype=np.float16)
    for eps in range(L):
        u[eps, 0] = u0/(1+ np.exp( (eps+1) - eps0))
    return u

def initializeU2(eps0, u0, L, t):
    u = np.zeros((L, t), dtype=np.float32)
    for eps in range(L):
        u[eps, 0] = u0*np.exp(-((eps+1) - eps0)**2)
    return u

def updateU(u, r, q, L, t):
    for eps in range(L):
        u[eps, t] = u[eps, t-1] + 0.1*(r*u[eps, t-1] - r/q*u[eps, t-1]**2 - u[eps, t-1]/(1+u[eps, t-1]))
        if eps !=  0 and eps != L-1:
            u[eps, t] = u[eps, t] + 0.1*(u[eps + 1 , t-1] + u[eps - 1, t-1] - 2*u[eps, t-1] )
    return u


def taskb():
    r =1/2
    q = 8
    L = 100
    tmax = 100
    #du/dtau = rho*u - rho/q *u**2 - u/(1+u ) + du^2/dxi^2
    u0_1 = -(1-q)/2 + np.sqrt(q- q/r + (1-q)**2/2) 
    #(r*(q - 1) + np.sqrt(r*(q**2*r + 2*q*r - 4*q + r)))/(2*r)
    u0_2 =  -(1-q)/2 - np.sqrt(q- q/r + (1-q)**2/2) 
    u0_3 = 1.1*u0_2
    
    eps0_1 = 20
    eps0_2 = 50
    eps0_3 = 50

    u_list = [u0_1, u0_2, u0_3]
    eps_list = [eps0_1, eps0_2, eps0_3]

    for i in range(3):
        curr_eps = eps_list[i]
        curr_u = u_list[i]
        u = initializeU(curr_eps, curr_u, L, tmax) 
    
    
        for t in range(1, tmax):
            u = updateU(u, r, q, L, t)
        
        
        plt.imshow(u, label = 't = 0')
        plt.ylabel('$\\xi$')
        plt.xlabel('$\\tau$')
        plt.colorbar()
        plt.show()
        for i in range(100):
            plt.plot(range(L),u[:,i], label = 't = ' + str(i*0.01))
        
        plt.show()

taskb()

def taskc():
    r =1/2
    q = 8
    L = 100
    tmax = 100
    #du/dtau = rho*u - rho/q *u**2 - u/(1+u ) + du^2/dxi^2
    u0_1 = -(1-q)/2 + np.sqrt(q- q/r + (1-q)**2/2) 
    eps0 = 50
    u_list = [u0_1, 3*u0_1]
   
    for i in range(1):
        u = initializeU2(eps0, u_list[i], L, tmax) 
        for t in range(1, tmax):
            u = updateU(u, r, q, L, t)
        print(u[:,1])
        plt.imshow(u)
        plt.title('$u(\\xi , \\tau)$, u0 = ' + str(np.round(u_list[i], 2)) )
        plt.ylabel('$\\xi$')
        plt.xlabel('$\\tau$')
        plt.colorbar()
        plt.show()
#taskc()

def test():
    arr = np.zeros((5,5))
    for i in range(5):
        for j in range(5):
            arr[i,j] = i+j
    print(arr)
    plt.imshow(arr)
    plt.colorbar()
    plt.show()

    
