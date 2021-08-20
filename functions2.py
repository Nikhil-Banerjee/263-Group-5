import pandas as pd
from os import sep
from functools import reduce
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def odePressure(t, P, q, a, b, P0):
    ''' Return the derivative dP/dt at a time, t for given parameters.

        Parameters:
        -----------
        t : float
            Independent variable.
        P : float
            Dependent varaible.
        a : float
            Source/sink strength parameter.
        b : float
            Recharge strength parameter.
        q : float
            Source/sink rate.
        P0 : float
            Initial Pressure values.

        Returns:
        --------
        dPdt : float
            Derivative of dependent variable with respect to independent variable.

        Notes:
        ------
        None

        Examples:
        ---------
        >>> 

    '''

    dPdt = -a*q - b*(P - P0)
    return dPdt

def odeTemp(t, T, P, T0, P0, Tsteam, a, b, c,q):
    ''' Return the derivative dT/dt at a time, t for given parameters.
        dT/dt = a*q*(Tsteam - T) - b(P - P0)(Tdash - T) - c(T - T0)

        Parameters:
        -----------
        t : float
            Independent variable.
        T : float
            Dependent varaible.
        P : float
            Pressure values.
        T0 : float
            initial Temperature
        P0 : float
            initial Pressure
        Tdash : float
            function returning values for T'(t)
        a : float
            superparameter 1.
        b : float
            superparameter 2.
        c: float
            superparameter 3.
        Returns:
        --------
        dTdt : float
            Derivative of dependent variable with respect to independent variable.

        Notes:
        ------
        None

        Examples:
        ---------
        >>> ADD EXAMPLES

    '''
    dTdt = a*q*(Tsteam - T) - b*(P - P0)*(Tprime(t, P, T, P0, T0) - T) - c*(T - T0)
    #dTdt = a*q*(Tsteam - T) - b*(P - P0)*T - c*(T - T0)

    return dTdt

def Tprime(t, P, T, P0, T0):
    ''' Return the current Temperature if current Pressure is more than initial Pressure, initial Temperature otherwise

        Parameters:
        -----------
        t : float
            current time
        P : float
            current Pressure
        T : float
            current Temperature
        P0 : float
            initial Pressure
        T0 : float
            initial Temperature
        
        Returns:
        --------
        Tprime : float
            Returns the required values for temperature depending on directin of flow.
    '''
    if (P > P0):
        return T
    else:  
        return T0


def loadGivenData():
    oil = pd.read_csv("data" + sep + "tr_oil.txt")
    pressure = pd.read_csv("data" + sep + "tr_p.txt")
    steam = pd.read_csv("data" + sep + "tr_steam.txt")
    temp = pd.read_csv("data" + sep + "tr_T.txt")
    water = pd.read_csv("data" + sep + "tr_water.txt")

    dataArray = [oil, pressure, steam, temp, water]
    dataArray = [df.set_index('days') for df in dataArray]

    data = reduce(lambda left, right: pd.merge(left, right, on = ['days'], how = 'outer'), dataArray).sort_index()

    return data

def loadData():
    oil = np.genfromtxt("data" + sep + "tr_oil.txt",delimiter=',',skip_header=1).T
    pressure = np.genfromtxt("data" + sep + "tr_p.txt",delimiter=',',skip_header=1).T
    steam = np.genfromtxt("data" + sep + "tr_steam.txt",delimiter=',',skip_header=1).T
    temp = np.genfromtxt("data" + sep + "tr_T.txt",delimiter=',',skip_header=1).T
    water = np.genfromtxt("data" + sep + "tr_water.txt",delimiter=',',skip_header=1).T

    oil=np.concatenate(([[0],[0]], oil),axis=1)
    water=np.concatenate(([[0],[0]], water),axis=1)
    steam=np.concatenate((steam,[[218],[0]]),axis=1)

    return oil, pressure, steam, temp, water


def interpolate(values,t):

    n=len(values[1])
    m=np.zeros(n-1)
    c=np.zeros(n-1)
    for i in range(n-1):
        # q=m*time+c
        m[i]=(values[1][i+1]-values[1][i])/(values[0][i+1]-values[0][i])
        if m[i]==float('inf'):
            m[i]=0
        c[i]=values[1][i]-m[i]*values[0][i]

    idx=0
    value=np.zeros(len(t))
    for i in range(len(t)-1):
        while not (t[i]>=values[0][idx] and t[i]<=values[0][idx+1]):
            idx+=1
        value[i]=m[idx]*t[i]+c[idx]
        
    return value

def  solve_ode(f, t, dt, x0, pars):
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
    q=q_term(t)
    # improved euler method
    for i in range(len(t)-1):
        k1=f(t[i],x1[i],q[i],*pars)
        x1[i+1]=k1*dt+x1[i]
        k2=f(t[i+1],x1[i+1],q[i+1],*pars)
        x[i+1]=dt*(k1+k2)*0.5+x1[i]
    return t,x  

def solve_tempode(f,t,dt,T0,p,a,b,c,p0):
    Tsteam=260
    oil, pressure, steam, temp, water=loadData()    
    q=interpolate(steam,t)   

    x1=np.zeros(len(t))
    x1[0]=T0
    x=np.zeros(len(t))
    x[0]=T0
    # improved euler method
    for i in range(len(t)-1):
        k1=f(t[i], x1[i], p[i], T0, p0, Tsteam, a, b, c, q[i])
        x1[i+1]=k1*dt+x1[i]
        k2=f(t[i+1], x1[i+1], p[i+1], T0, p0, Tsteam, a, b, c,q[i+1])
        x[i+1]=dt*(k1+k2)*0.5+x1[i]

    return x


def q_term(t):
    oil, pressure, steam, temp, water=loadData()    
    oil1=interpolate(oil,t)
    water1=interpolate(water,t)
    steam1=interpolate(steam,t)
    return (water1-steam1+oil1)

def fit_pressure(t,a,b,p0):
    t,p=solve_ode(odePressure,t,t[1]-t[0],p0,[a,b,p0])
    return p

def fit_temp(t, a, b, c, T0, P0):
    #T0=170
    # p=solve_ode(odePressure,t,t[1]-t[0],p0,ParsdP)
    
    x=solve_tempode(odeTemp,t,t[1]-t[0],T0,p[1],a,b,c,P0)
    return x    

if __name__ == "__main__":
    data = loadGivenData()
    oil, pressure, steam, temp, water=loadData()

    t=np.linspace(0,217,1000)

    # Initial guesses for pressure parameters.
    a, b = 0.2, 0.05
    p0 = pressure[1][0]

    parsFoundP, _ = curve_fit(fit_pressure, pressure[0], pressure[1],[a, b, p0])

    p = solve_ode(odePressure, t, t[1] - t[0], p0, parsFoundP)

    pressureMisfit = interpolate(p, pressure[0]) - pressure[1]
    
    f,axe = plt.subplots(1,2)
    axe[0].plot(t,p[1],'k--',label="a = {:.2f}\nb = {:.2f}\nP0 = {:.2f}".format(parsFoundP[0], parsFoundP[1], parsFoundP[2]))
    axe[0].plot(pressure[0],pressure[1],'r.',label='data')
    axe[1].plot(pressure[0],pressureMisfit,'kx')
    axe[1].plot(pressure[0],np.zeros(len(pressureMisfit)),'r--')
    axe[0].set_ylabel('Pressure [kPa]')
    axe[0].set_xlabel('time [s]')
    axe[0].set_title('Comparison of model to observed \n pressure value')
    axe[0].legend(loc='upper right',prop={'size': 7})
    axe[1].set_ylabel('pressure misfit [kpa]')
    axe[1].set_xlabel('time [s]')
    axe[1].set_title('Best fit LMP model')
    

    Tsteam = 260
    # Initial guesses for temperature parameters.
    T0 = temp[1][0]
    a,b,c=0.0001,0.0001,0.01

    parsFoundT, _ = curve_fit(lambda t,a,b,c,T0: fit_temp(t,a,b,c,T0,parsFoundP[2]), temp[0], temp[1],[a,b,c,T0])
    print(parsFoundT)

    x=solve_tempode(odeTemp, t, t[1] - t[0], parsFoundT[3], p[1], a = parsFoundT[0],b = parsFoundT[1],c = parsFoundT[2], p0 = parsFoundP[2])

    temp_misfit=interpolate([t,x],temp[0])-temp[1]
    
    f2,axe2 = plt.subplots(1,2)
    axe2[0].plot(t,x,'k--',label='a = {parsFoundT[0]:.2f}\nb = {parsFoundT[1]:.2f}\nc = {parsFoundT[2]:.2f}')
    axe2[0].plot(temp[0],temp[1],'r.',label='data')
    axe2[1].plot(temp[0],temp_misfit,'kx')
    axe2[1].plot(temp[0],temp[0]*0,'r--')
    axe2[0].set_ylabel('temperature [°C]')
    axe2[0].set_xlabel('time [s]')
    axe2[0].set_title('Comparison of model to observed \n temperature value')
    axe2[0].legend(loc='upper right',prop={'size': 7})
    axe2[1].set_ylabel('temperature misfit [°C]')
    axe2[1].set_xlabel('time [s]')
    axe2[1].set_title('Best fit LMP model')

    plt.show()
    