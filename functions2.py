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


def odeSolnAnalytic(t):

    soln = np.exp(-t)
    return soln


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

def q_term(t):
    oil, pressure, steam, temp, water=loadData()    
    oil1=interpolate(oil,t)
    water1=interpolate(water,t)
    steam1=interpolate(steam,t)
    return (water1-steam1+oil1)

def qs_scenario1(t):
    # Steam injection of 1000 tonnes per day for 60 days.
    q = np.where(t <= 280.28, 1000, 0)
    return q
    

def q_scenario1(t):
    # Steam injection of 1000 tonnes per day for 60 days, followed by 90 day production periods.
    q = np.where(t <= 280.28, -1000, 80.637)
    return q

def qs_scenario2(t):
    # No steam injection at all.
    return np.zeros(len(t))

def q_scenario2(t):
    # No steam injection at all.
    return np.ones(len(t)) * 80.637

def qs_scenario3(t):
    # Current steam injection rate for 60 days, followed by 90 day production periods.
    q = np.where(t <= 280.28, 460, 0)
    return q

def q_scenario3(t):
    q = np.where(t <= 280.28, -460, 80.637)
    return q

def qs_scenario4(t):
    # Steam injection of 2000 tonnes per day for 60 days.
    q = np.where(t <= 280.28, 2000, 0)
    return q


def q_scenario4(t):
    # Steam injection of 2000 tonnes per day for 60 days, followed by 90 day production periods.
    q = np.where(t <= 280.28, -2000, 80.637)
    return q


def  solve_ode(f, t, dt, x0, pars, qmodel = q_term):
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
    q = qmodel(t)

    # improved euler method
    for i in range(len(t)-1):
        k1=f(t[i],x1[i],q[i],*pars)
        x1[i+1]=k1*dt+x1[i]
        k2=f(t[i+1],x1[i+1],q[i+1],*pars)
        x[i+1]=dt*(k1+k2)*0.5+x1[i]
    return t,x  

def qs(t):
    oil, pressure, steam, temp, water=loadData()  

    return interpolate(steam,t)


def solve_tempode(f,t,dt,T0,p,a,b,c,p0, qmodel = qs):
    Tsteam=260 
    q = qmodel(t) 

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

    t1=np.linspace(0,217,1000)

    # Pressure model fitting
    # Initial guesses for pressure parameters.
    a, b = 0.2, 0.05
    p0 = pressure[1][0]

    parsFoundP, _ = curve_fit(fit_pressure, pressure[0], pressure[1],[a, b, p0])

    p = solve_ode(odePressure, t1, t1[1] - t1[0], p0, parsFoundP)

    pressureMisfit = interpolate(p, pressure[0]) - pressure[1]
    
    f1,ax1 = plt.subplots(1,2)
    ax1[0].plot(t1,p[1],'k--',label="a = {:.2f}\nb = {:.2f}\nP0 = {:.2f}".format(parsFoundP[0], parsFoundP[1], parsFoundP[2]))
    ax1[0].plot(pressure[0],pressure[1],'r.',label='data')
    ax1[1].plot(pressure[0],pressureMisfit,'kx')
    ax1[1].plot(pressure[0],np.zeros(len(pressureMisfit)),'r--')
    ax1[0].set_xlim([0, t1[-1]])
    ax1[1].set_xlim([0, t1[-1]])
    ax1[0].set_ylabel('Pressure [Pa]')
    ax1[0].set_xlabel('time [days]')
    ax1[0].set_title('Comparison of model to observed pressure value')
    ax1[0].legend(loc='upper right',prop={'size': 7})
    ax1[1].set_ylabel('pressure misfit [Pa]')
    ax1[1].set_xlabel('time [days]')
    ax1[1].set_title('Best fit LMP model')
    
    # Temperature model fitting
    Tsteam = 260
    # Initial guesses for temperature parameters.
    T0 = temp[1][0]
    a,b,c = 0.0001, 0.0001, 0.01

    parsFoundT, _ = curve_fit(lambda t,a,b,c,T0: fit_temp(t,a,b,c,T0,parsFoundP[2]), temp[0], temp[1],[a,b,c,T0])
    print(parsFoundT)

    T=solve_tempode(odeTemp, t1, t1[1] - t1[0], parsFoundT[3], p[1], a = parsFoundT[0],b = parsFoundT[1],c = parsFoundT[2], p0 = parsFoundP[2])

    temp_misfit=interpolate([t1,T],temp[0])-temp[1]
    
    f2,ax2 = plt.subplots(1,2)
    ax2[0].plot(t1,T,'k--',label='a = {:3f}\nb = {:3f}\nc = {:3f}'.format(parsFoundT[0],parsFoundT[1],parsFoundT[2]))
    ax2[0].plot(temp[0],temp[1],'r.',label='data')
    ax2[0].plot(t1, np.ones(len(t1)) * 240, 'g-', label = 'Toxic contaminant dissociation temperature')
    ax2[1].plot(temp[0],temp_misfit,'kx')
    ax2[1].plot(temp[0],temp[0]*0,'r--')
    ax2[0].set_xlim([0, t1[-1]])
    ax2[1].set_xlim([0, t1[-1]])
    ax2[0].set_ylabel('temperature [°C]')
    ax2[0].set_xlabel('time [days]')
    ax2[0].set_title('Comparison of model to observed \n temperature value')
    ax2[0].legend(loc='lower left',prop={'size': 7})
    ax2[1].set_ylabel('temperature misfit [°C]')
    ax2[1].set_xlabel('time [days]')
    ax2[1].set_title('Best fit LMP model')

    # Testing the numerical solver by comparing to a known analytical solution.
    tn1 = np.linspace(0,10,150)
    tnum1, numericP1 = solve_ode(odePressure, tn1, tn1[1]-tn1[0], 1, [0,1,0])
    analyticP1 = odeSolnAnalytic(tn1)

    tn2 = np.linspace(0,10,5)
    tnum2, numericP2 = solve_ode(odePressure, tn2, tn2[1]-tn2[0], 1, [0,1,0])
    analyticP2 = odeSolnAnalytic(tn2)


    f3, ax3 = plt.subplots(1,2)
    ax3[0].plot(tnum1, numericP1, 'kx', label = 'numeric solution')
    ax3[0].plot(tn1, analyticP1, 'b-', label = 'analytical solution')
    ax3[0].plot(tn1, np.zeros(len(tn1)), 'b--', label = 'steady-state')
    ax3[0].legend()
    ax3[0].set_title('benchmark: a=0.00, b=1.00, P0=0.00')
    ax3[0].set_ylabel('Pressure (Pa)')
    ax3[0].set_xlabel('time (days)')

    ax3[1].plot(tnum2, numericP2, 'kx', label = 'numeric solution')
    ax3[1].plot(tn2, analyticP2, 'b-', label = 'analytical solution')
    ax3[1].plot(tn2, np.zeros(len(tn2)), 'b--', label = 'steady-state')
    ax3[1].legend()
    ax3[1].set_title('Instability at large time-step')
    ax3[1].set_ylabel('Pressure (Pa)')
    ax3[1].set_xlabel('time (days)')

    # Forecasts

    # Forecast 1
    # Tood Energy proposal of steam injection of 1000 tonnes per day 60 days, followed by 90 day production periods.
    ts = np.linspace(t1[-1], 370.28, 100)
    FP1 = solve_ode(odePressure, ts, ts[1] - ts[0], p[1][-1], parsFoundP, q_scenario1) 
    FT1 = solve_tempode(odeTemp, ts, ts[1] - ts[0], parsFoundT[3], FP1[1], parsFoundT[0], parsFoundT[1], parsFoundT[2], parsFoundP[2], qs_scenario1)

    # Forecast 2
    # No steam injection
    FP2 = solve_ode(odePressure, ts, ts[1] - ts[0], p[1][-1], parsFoundP, q_scenario2)
    FT2 = solve_tempode(odeTemp, ts, ts[1] - ts[0], parsFoundT[3], FP2[1], parsFoundT[0], parsFoundT[1], parsFoundT[2], parsFoundP[2], qs_scenario2)

    # Forecast 3
    # Current steam injection of 460 tonnes per day for 60 days, followed by 90 day production periods.
    FP3 = solve_ode(odePressure, ts, ts[1] - ts[0], p[1][-1], parsFoundP, q_scenario3)
    FT3 = solve_tempode(odeTemp, ts, ts[1] - ts[0], parsFoundT[3], FP3[1], parsFoundT[0], parsFoundT[1], parsFoundT[2], parsFoundP[2], qs_scenario3)

    # Forecast 4 
    # steam injection of 2000 tonnes per day 60 days, followed by 90 day production periods.
    FP4 = solve_ode(odePressure, ts, ts[1] - ts[0], p[1][-1], parsFoundP, q_scenario4)
    FT4 = solve_tempode(odeTemp, ts, ts[1] - ts[0], parsFoundT[3], FP4[1], parsFoundT[0], parsFoundT[1], parsFoundT[2], parsFoundP[2], qs_scenario4)

    # Forecast plot (only shows temperature plot)
    tOverall = np.arange(t1[0],ts[-1])
    f4, ax4 = plt.subplots(1,1)
    ax4.plot(t1,T,'b-',label='Model')
    ax4.plot(temp[0],temp[1],'ko',label='data')
    ax4.plot(ts, FT4, color = '#8B008B', ls = '-', label = 'Steam injection = 2000 t/d')
    ax4.plot(ts, FT1, 'y-', label = 'Todd Energy proposed steam injection = 1000 t/d')
    ax4.plot(ts, FT3, color = '#00FFFF', ls = '-', label = 'Current steam injection = 460 t/d')
    ax4.plot(ts, FT2, 'g-', label = 'Steam injection = 0 t/d')
    ax4.plot(tOverall,np.ones(len(tOverall)) * 240,'r-', label =  'Toxic contaminant dissociation temperature')
    ax4.legend()
    ax4.text(x = 225, y = 130, s = 'All forecasted injection phases are 60 days followed by 90 day production periods.', bbox = dict(facecolor='none', edgecolor='black', pad=5.0))
    ax4.set_xlim([t1[0], ts[-1]])
    ax4.set_xlabel('time (days)')
    ax4.set_ylabel('Temperature (°C)')





    plt.show()
    