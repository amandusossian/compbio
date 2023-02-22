import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import odeint

def taska():
    q_val = 8
    rho = 0.5
    r, q, u = sp.symbols('r q u')
    fp= sp.solve(r*u*(1-u/q)-u/(1+u), u)
    u_star1_val = fp[0].subs([(q, q_val), (r, rho)])
    u_star2_val = fp[1].subs([(q, q_val), (r, rho)])
    u_star3_val = fp[2].subs([(q, q_val), (r, rho)])
    
    print(u_star1_val)
    print(u_star2_val)
    print(u_star3_val)

#taska()

def initializeU(eps0, u0, L, t):
    u = np.zeros((L, t), dtype = np.float64)
    eps = np.arange(1, L+1)
    u[:, 0] = u0/(np.ones(L)+ np.exp( eps - eps0))
    print(u[:,0])
    return u

def initializeU2(eps0, u0, L, t):
    u = np.zeros((L, t))
    eps = np.arange(1, L+1)
    u[:, 0] = u0*np.exp(-np.power(eps - eps0,2))
    return u
"""
def updateU(old_u, r, q, L, t):
    time_step = 0.01
    for eps in range(L):
        new_u[1:-2] = old_u[eps1] + time_step*(r*u[eps, t-1] - r/q*u[eps, t-1]**2 - u[eps, t-1]/(1+u[eps, t-1]))
        if eps !=  0 and eps != L-1:
            new_u[eps, t] = u[eps, t] + time_step*(u[eps + 1 , t-1] + u[eps - 1, t-1] - 2*u[eps, t-1] )
        elif eps==0:
            new_u[eps] = u[eps, t] + time_step*(u[1, t-1] - u[0, t-1])
        else: 
            new_u[eps] = u[eps, t] + time_step*(u[eps-1, t-1] - u[eps, t-1])
    return old_u
"""   

def updateU(u, r, q, L, tmax):
    time_step = 0.4
    new_u = np.zeros(L)
    for t in range(1, tmax):
        new_u = u[:, t-1] + time_step*(r*u[:, t-1]*(1-u[:, t-1]/q) - u[:, t-1]/(1+u[:, t-1]))
        new_u[1:-1] = new_u[1:-1] + time_step*(u[2:, t-1] + u[:-2, t-1] - 2*u[1:-1, t-1] )
        new_u[0] = new_u[0] + time_step*(u[1, t-1] - u[0, t-1])
        new_u[L-1] = new_u[L-1] + time_step*(u[L-2, t-1] - u[L-1, t-1])
        u[:,t] = new_u
    return u


def taskb():
    r =0.5
    q = 8
    L = 100
    tmax = 500
  
    u0_1 = -(1- q)/2 + np.sqrt(((1- q)/2)**2 - (q/r-q))
    u0_2 =  -(1- q)/2 - np.sqrt(((1- q)/2)**2 - (q/r-q))
   
    # Actual values
    # u0_1 = 5.56155281280883
    # u0_2 = 1.43844718719117
    
    eps0_1 = 20
    eps0_2 = 50
    eps0_3 = 50

    u_list = [u0_1, u0_2, 1.1*u0_2]
    eps_list = [eps0_1, eps0_2, eps0_3]

    for i in range(1):
        curr_eps = eps_list[i]
        curr_u = u_list[i]

        u = initializeU(curr_eps, curr_u, L, tmax)  
        u = updateU(u, r, q, L, tmax)
        
        if False:
            plt.imshow(u)
            plt.ylabel('$\\xi$')
            plt.xlabel('$\\tau$')
            plt.colorbar()
            plt.show()
        for i in np.arange(1, tmax):
            plt.plot(u[:,i], label = 't = ' + str(i))
            plt.ylabel('u')
            plt.xlabel('$\\xi$')
            #plt.legend()
        
        plt.show()

taskb()

def taskc():
    r =1/2
    q = 8
    L = 100
    tmax = 500
    #du/dtau = rho*u - rho/q *u**2 - u/(1+u ) + du^2/dxi^2
    u0_1 = -(1-q)/2 + np.sqrt(q- q/r + (1-q)**2/2) 
    eps0 = 50
    u_list = [u0_1, 3*u0_1]
   
    for i in range(1):
        u = initializeU2(eps0, u_list[i], L, tmax) 
        for t in range(1, tmax):
            u[t] = updateU(u, r, q, L, t)
        print(u[:,1])
        plt.imshow(u)
        plt.title('$u(\\xi , \\tau)$, u0 = ' + str(np.round(u_list[i], 2)) )
        plt.ylabel('$\\xi$')
        plt.xlabel('$\\tau$')
        plt.colorbar()
        plt.show()
#taskc()

def test():
    a =np.array([1,2,3,4])
    print(a[1:-1])

#test()
