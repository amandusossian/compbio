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

def initializeU(eps0, u0, L, tmax, time_step):
    u = np.zeros((L, tmax), dtype = np.float64)
    eps = np.arange(1, L+1)
    u[:, 0] = u0/(np.ones(L)+ np.exp( eps - eps0))
    
    return u

def initializeU2(eps0, u0, L, t):
    u = np.zeros((L, t))
    eps = np.arange(1, L+1)
    u[:, 0] = u0*np.exp(-np.power(eps - eps0,2))
    return u
   

def updateU(u, r, q, L, tmax, time_step):
    new_u = np.zeros(L)
    for t in range(1, tmax):
        new_u = u[:, t-1] + time_step*(r*u[:, t-1]*(1-u[:, t-1]/q) - u[:, t-1]/(1+u[:, t-1]))
        new_u[1:-1] = new_u[1:-1] + time_step*(u[2:, t-1] + u[:-2, t-1] - 2*u[1:-1, t-1] )
        new_u[0] = new_u[0] + time_step*(u[1, t-1] - u[0, t-1])
        new_u[L-1] = new_u[L-1] + time_step*(u[L-2, t-1] - u[L-1, t-1])
        u[:,t] = new_u
    return u

def EstimateC(u, L, t_search, time_step, direction):
    u_mid = np.max(u[:, t_search])/2
    if direction == 'f':
        x1 = np.where(u[:, t_search] < u_mid )[0][0]                # closest to mid point from underneath
        delta_t =  np.where(u[x1+1, t_search:] > u_mid )[0][0] # time for next to rise
        print('c = ' + str(1/(time_step*delta_t) ))
    else: 
        pass
    return 1/(time_step*delta_t)
    
# Wave velocity = 0.12121212121212122
# Wave velocity = 1.639344262295082
# Wave velocity = 0.09250693802035152

def CreateUPhase(u, L, tmax, t_plot, time_step, c, r, q     ):
    curr_u = u[:, t_plot]
    next_u = u[:, t_plot+1]
    u_z = u[:, t_plot]- np.ones(L)*c*t_plot*time_step
    
    x = np.linspace(0,6,100)
    y = np.linspace(-2,2,100)
    x, y = np.meshgrid(x,y)
    Xdot = y
    Ydot = -c*y-(r*x*(1-x/q)-x/(1+x))

    fig, axs = plt.subplots(1,1, figsize=(10,5))
    axs.streamplot(x, y,  Xdot, Ydot, density=1)
    t = np.arange(0, 20, 0.01)
    v_init = np.array(curr_u[1:] - curr_u[:-1])
    v_init = np.append(v_init, 0)
    for i in range(L):
        start_point =  [curr_u[i],v_init[i]]
        result = odeint(odefun1, [start_point[0], start_point[1]], t, args = (c, r, q), tfirst=True)
        axs.plot(result[:,0], result[:,1], 'r', label = 'Trajectory')
    axs.set_xlim([0,6])
    axs.set_ylim([-2,2])
    plt.plot()
    plt.show()
    return axs

def odefun1(t, state,c, r, q):
    x, y = state
    Xdot = y
    Ydot = -c*y-(r*x*(1-x/q)-x/(1+x))
    return [Xdot, Ydot]


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
    time_step = 0.4
    for i in [0]:
        if i==1:
            direction = 'b'
        else:
            direction = 'f'
        curr_eps = eps_list[i]
        curr_u = u_list[i]

        u = initializeU(curr_eps, curr_u, L, tmax, time_step)  
        u = updateU(u, r, q, L, tmax, time_step)
        t_search = 250
        c = EstimateC(u, L, t_search, time_step, direction)

        t_plot = 250
        diff = np.diff(u[:,t_plot])
        #u_phase = CreateUPhase(u, L, tmax, t_plot, time_step, c, r, q)
        #print('Wavenumber = ' + str(np.round(c,3)))
        fig, axs = plt.subplots(1, 2)
        #pos1 = axs[1].imshow(u_phase, aspect="auto", cmap='jet', extent=[0, tmax, 0, L])
        axs[1].plot(diff, u[:-1, t_plot])
        #axs[1].set_title('u($\\xi$, $\\tau$)')

        axs[1].set_ylabel('$u$')
        axs[1].set_xlabel('$du/d\\xi$')
        #fig.colorbar(pos1, ax = axs[1])
        axs[1].grid()
        axs[0].plot(u[:,t_plot])
        axs[0].set_ylabel('u')
        axs[0].set_xlabel('$\\xi$')
        axs[0].grid()
        axs[0].set_title('u($\\xi$, $\\tau$), $u_0$ = ' + str(np.round(curr_u, 2)) + ', $\\xi_0$ = ' + str(np.round(curr_eps)) + ', $\\tau$ = ' + str(t_plot))
        fig.suptitle('First case, $u_0$ = $u_+$, $\\xi_0$ = 20')
        
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
