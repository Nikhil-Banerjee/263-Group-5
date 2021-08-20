from functions import*
import numpy as np
from matplotlib import pyplot as plt

#simple LMP for energy which does not have net mass flow(kettle experiment)
def simple_odeTem(t,T,T0,a,b,q):
    return a*q-b*(T-T0)

def benchmark():
    oil, pressure, steam, temp, water=loadData()
    #simplified scenario where initial t0,p0,constants, net mass = 1 with base unit
    p0,T0,a,b=1,1,1,1
    pars=a,b,p0
    t=np.linspace(0,10,1000)
    t_sol=np.linspace(0,10,16)
    q=np.zeros(len(t))+1

    #improve euler solution for pressure
    pi=solve_ode(odePressure,t,t[1]-t[0],p0,pars,q)
    
    #analytical solution for pressure
    p_sol=p0-a*1/b*(1-np.e**(-b*t_sol))

    #improve euler solution for temperature
    Ti=solve_ode(simple_odeTem,t,t[1]-t[0],T0,pars,q)
    #analytical solution for temperature
    T_sol=a*1/b*(1-np.e**(-b*t_sol))+T0

    f,axe = plt.subplots(1,2)
    axe[0].plot(t,pi[1],'k--',label='Numerical Solutions')
    axe[0].plot(t_sol,p_sol,'r.',label='Analytical Solution')
    axe[1].plot(t,Ti[1],'k--',label='Numerical Solutions')
    axe[1].plot(t_sol,T_sol,'r.',label='Analytical Solution')
    #label
    axe[0].set_ylabel('Pressure [Pa]')
    axe[0].set_xlabel('time [s]')
    axe[0].set_title('Benchmark of the Analytical and Numerical \n Solutions of the pressure ODE')
    axe[0].legend(loc='upper right',prop={'size': 7})
    axe[1].set_ylabel('Temperature [°C]')
    axe[1].set_xlabel('time [s]')
    axe[1].set_title('Benchmark of the Analytical and Numerical \n Solutions of the temperature ODE')
    axe[1].legend(loc='lower right',prop={'size': 7})
    #plt.show()


    inv_h=np.linspace(5,95,19)
    stepsize=1/inv_h
    sol=np.zeros(len(inv_h))
    j=0
    for h in stepsize:
        t=np.arange(0,10,h)
        pi=solve_ode(odePressure,t,t[1]-t[0],p0,pars,q)
        sol[j]=pi[1][-1]
        j+=1
        
    f,axe = plt.subplots(1)
    axe.plot(inv_h,sol,'r.')
    axe.set_ylabel('Pressure [Pa]')
    axe.set_xlabel('1/h')
    axe.set_title('Convergence test for Improved Euler')
    axe.set_ylim([1.7e-5, 20e-5])
    plt.show()


benchmark()