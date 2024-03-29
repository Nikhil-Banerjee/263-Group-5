from main import*
import numpy as np
from matplotlib import pyplot as plt

#simple pressure and temperature ODEs
def simpleodePressure(t, P, q, a, b, P0):
    ''' Returns dP/dt

    Parameters:
    -----------
    t : float
        Independent time variable.
    P : float
        Dependent Pressure variable.
    a : float
        Source/sink strength parameter.
    b : float
        Recharge strength parameter.
    q : function 
        Source/sink rate function
    P0 : float
        Initial Pressure value
  
    Returns:
    --------
    dPdt : float
           Derivative of Pressure with respect to time.
    '''
    dPdt = -a*q - b*(P - P0)
    return dPdt
def simple_odeTem(t,T,T0,a,b,q):
    '''Returns dT/dt

    Parameters:
    -----------
    t : float
        Independent time variable.
    T : float
        Dependent Temperature variable.
    q : function
        Flow rate of steam - qs(t)
    a : float
        extraction/injection parameter
    b : float
        recharge parameter
    T0 : float
        initial Temperature
    
    Returns:
    --------
    dTdt : float
        Derivative of Temperature with respect to time.
    '''
    return a*q-b*(T-T0)
def solve_simpleode(f, t, dt, x0, pars,q):
    '''solve ODE numerically with forcing term is altering
    Parameters:
        -----------
        f : callable
            Function that returns dxdt given variable and parameter inputs.
        t0 : float
            Initial time of solution.
        t1 : float
            Final time of solution.
        dt : float
            Time step length.
        x0 : float
            Initial value of solution.
        pars : array-like
            List of parameters passed to ODE function f.
    '''
    x1=np.zeros(len(t))
    x1[0]=x0
    x=np.zeros(len(t))
    x[0]=x0
    
    # improved euler method
    for i in range(len(t)-1):
        k1=f(t[i],x1[i],q[i],*pars)
        x1[i+1]=k1*dt+x1[i]
        k2=f(t[i+1],x1[i+1],q[i+1],*pars)
        x[i+1]=dt*(k1+k2)*0.5+x1[i]
    return t,x    
def benchmark():
    oil, pressure, steam, temp, water=loadData()
    #simplified scenario where initial t0,p0,constants, net mass = 1 with base unit
    p0,T0,a,b=1,1,1,1
    pars=a,b,p0
    t=np.linspace(0,10,1000)
    t_sol=np.linspace(0,10,16)
    q=np.zeros(len(t))+1

    #improve euler solution for pressure
    pi=solve_simpleode(simpleodePressure,t,t[1]-t[0],p0,pars,q)
    
    #analytical solution for pressure
    p_sol=p0-a*1/b*(1-np.e**(-b*t_sol))

    #improve euler solution for temperature
    Ti=solve_simpleode(simple_odeTem,t,t[1]-t[0],T0,pars,q)
    #analytical solution for temperature
    T_sol=a*1/b*(1-np.e**(-b*t_sol))+T0

    f,axe = plt.subplots(1,2)
    axe[0].plot(t,pi[1],'k--',label='Numerical Solutions')
    axe[0].plot(t_sol,p_sol,'r.',label='Analytical Solution')
    axe[1].plot(t,Ti[1],'k--',label='Numerical Solutions')
    axe[1].plot(t_sol,T_sol,'r.',label='Analytical Solution')
    #label
    axe[0].set_ylabel('Pressure [kPa]')
    axe[0].set_xlabel('time [s]')
    axe[0].set_title('Benchmark of the Analytical and Numerical \n Solutions of the pressure ODE')
    axe[0].legend(loc='upper right',prop={'size': 7})
    axe[1].set_ylabel('Temperature [°C]')
    axe[1].set_xlabel('time [s]')
    axe[1].set_title('Benchmark of the Analytical and Numerical \n Solutions of the temperature ODE')
    axe[1].legend(loc='lower right',prop={'size': 7})

    #convergence test
    inv_h=np.linspace(5,95,19)
    stepsize=1/inv_h
    sol=np.zeros(len(inv_h))
    j=0
    for h in stepsize:
        t=np.arange(0,10,h)
        pi=solve_simpleode(simpleodePressure,t,t[1]-t[0],p0,pars,q)
        sol[j]=pi[1][-1]
        j+=1
        
    f,axe = plt.subplots(1)
    axe.plot(inv_h,sol,'r.', label = 'Numerical Solution')
    axe.plot(inv_h,-(1-np.exp(-10))+1 + np.zeros(len(inv_h)),'k--', label = 'Analytical Solution')
    axe.set_ylabel('Pressure [kPa]')
    axe.set_xlabel('1/h')
    axe.set_title('Convergence test for Improved Euler')
    axe.set_ylim([0, 70e-5])
    axe.set_xlim([0, 95])
    axe.annotate("", xy=(10, 0.000048), xytext=(10, 0.000088),arrowprops=dict(arrowstyle="->"))
    axe.legend()
