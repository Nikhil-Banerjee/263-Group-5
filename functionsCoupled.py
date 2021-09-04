import pandas as pd
from os import sep
from functools import reduce
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import fmin_slsqp
import random

Tsteam = 260

def odePressure(t, X, q, qs, a, b, c, M, P0, Y0):
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
    P, T = X
    dPdt = -a*q(t) - b*(P - P0)
    return dPdt

def odeTemp(t, X, q, qs, a, b, c, M, P0, T0):
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
            extraction/injection parameter
        b : float
            recharge parameter
        c: float
            conduction parameter
        M : float
            initial mass

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
    P, T = X
    dTdt = qs(t)/M*(Tsteam - T) - b/(a*M)*(P - P0)*(Tprime(t, P, T, P0, T0) - T) - c*(T - T0)
    #dTdt = qs/M*(Tsteam - T) - b/(a*M)*(P - P0)*T - c*(T - T0)
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

def odes(t, X, q, qs, a, b, c, M, P0, T0):
    dPdt = odePressure(t, X, q, qs, a, b, c, M, P0, T0)
    dTdt = odeTemp(t, X, q, qs, a, b, c, M, P0, T0)

    return [dPdt, dTdt]

def solve_odes(t, q, qs, pars):

    P0 = pars[-2]
    T0 = pars[-1]

    funWithQs = lambda t, X, a, b, c, M, P0, T0: odes(t, X, q, qs, a, b, c, M, P0, T0)

    soln = solve_ivp(funWithQs, [t[0], t[-1]], [P0, T0], t_eval = t, args=pars)

    P = soln.y[0]
    T = soln.y[1]

    return soln.t, P, T

def fit_odes(t, q, qs, pars):

    oil, pressure, steam, temp, water = loadData()

    tsol, P, T = solve_odes(t, q, qs, pars)

    # Here we are saying that we deem a 1°C error to be the same as 1 Pa of error.
    tempMultipler = 250
    pressureMultiplier = 1 # because units of pressure are given in kPa.

    # RSS = tempMultipler*np.sum((temp[1] - interpolate([t,T], temp[0]))**2) \
    #     + pressureMultiplier*np.sum((pressure[1] - interpolate([t,P], pressure[0]))**2)

    RSS = tempMultipler*np.mean(np.square(temp[1] - np.interp(temp[0], tsol, T))) \
        + pressureMultiplier*np.mean(np.square(pressure[1] - np.interp(pressure[0], tsol, P)))
    
    return RSS

def q_term(t):
    oil, pressure, steam, temp, water=loadData()    
    
    interpWater = np.interp(t, water[0], water[1])
    interpOil = np.interp(t, oil[0], oil[1])
    interpSteam = np.interp(t, steam[0], steam[1])

    q = interpWater - interpSteam + interpOil

    return q

def qs_term(t):
    oil, pressure, steam, temp, water=loadData() 
    return np.interp(t, steam[0], steam[1])

class Qterms:
    # Initializing the function by using default constructor.
    def __init__(self):
        oil, pressure, steam, temp, water=loadData()  
        
        self.oil = oil
        self.pressure = pressure
        self.steam = steam
        self.temp = temp
        self.water = water

        self.interpWater = interp1d(water[0], water[1])
        self.interpOil = interp1d(oil[0], oil[1])
        self.interpSteam = interp1d(steam[0], steam[1])

    def giveQ(self, t):
        # Interpolates the water function and extrapolates when required.
        if (t < self.water[0][0]):
            w = self.water[1][0]
        elif (t > self.water[0][-1]):
            w = self.water[1][-1]
        else:
            w = self.interpWater(t)
    
        # Interpolates the oil function and extrapolates when required.
        if (t < self.oil[0][0]):
            o = self.oil[1][0]
        elif (t > self.oil[0][-1]):
            o = self.oil[1][-1]
        else:
            o = self.interpOil(t)
        
        # Interpolates the steam function and extrapolates when required.
        s = self.giveQs(t)

        return w-s+o

    def giveQs(self, t):
        # Interpolates the steam function and extrapolates when required.
        if (t < self.steam[0][0]):
            s = self.steam[1][0]
        elif (t > self.steam[0][-1]):
            s = self.steam[1][-1]
        else:
            s = self.interpSteam(t)

        return s

""" Custom step-function """
class RandomDisplacementBounds(object):
    """random displacement with bounds:  see: https://stackoverflow.com/a/21967888/2320035
        Modified! (dropped acceptance-rejection sampling for a more specialized approach)
    """
    def __init__(self, xmin, xmax, stepsize=0.5):
        self.xmin = xmin
        self.xmax = xmax
        self.stepsize = stepsize

    def __call__(self, x):
        """take a random step but ensure the new position is within the bounds """
        min_step = np.maximum(self.xmin - x, -self.stepsize)
        max_step = np.minimum(self.xmax - x, self.stepsize)

        random_step = np.random.uniform(low=min_step, high=max_step, size=x.shape)
        xnew = x + random_step

        return xnew

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

if __name__ == "__main__":

    oil, pressure, steam, temp, water=loadData()

    t = np.linspace(0, 217, 1000)

    # Initial guesses for parameters:
    # a = 1.918e-1
    # b = 6.185e-2
    # c = 8.842e-2
    # M = 4.325e+3
    # P0 = 8.731e+2
    # T0 = 1.587e+2

    a = 0.2
    b = 0.05
    c = 0.5
    M = 5000
    P0 = pressure[1][0]
    T0 = temp[1][0]

    initGuess = (a,b,c,M,P0,T0)

    # Initializing the Qterms class:
    Q = Qterms()

    funcFit_odes = lambda pars: fit_odes(t,Q.giveQ, Q.giveQs, pars)

    parsMin = [1e-10,1e-10,1e-10,500,750,140]
    parsMax = [1,1,100,20000,1500,190]
    bnds = [(low, high) for low, high in zip(parsMin, parsMax)]

    # bnds = ((1e-13,1),(1e-13,1),(1e-13,1),(1,np.inf),(600,1500),(120,230))

    np.random.seed(99)
    # pars = basinhopping(funcFit_odes, initGuess, niter=200, disp = True,\
        #  minimizer_kwargs = {'method' : 'Nelder-Mead', 'bounds' : bnds}, \
        #  take_step=RandomDisplacementBounds(parsMin, parsMax))
    # pars = fmin_slsqp(funcFit_odes, initGuess, bounds = bnds)
    pars = minimize(funcFit_odes, initGuess, method = 'SLSQP', bounds = bnds)
    # Fix q functions to make them faster!!!
    # least sq function probably not gonna work because residuals are not squared and still being summed (cancellation may occur!!!)
    print(pars)


    tsol, P, T = solve_odes(t, Q.giveQ, Q.giveQs, pars.x)
    tempMisfit = temp[1] - np.interp(temp[0], tsol, T)
    # tempMSE = np.square(temp[1] - np.interp(temp[0], tsol, T))
    pressureMisfit = pressure[1] - np.interp(pressure[0], tsol, P)

    f1,ax1 = plt.subplots(1,2)
    ax1[0].plot(tsol,P,'k--')
    ax1[0].plot(pressure[0],pressure[1],'r.',label='data')
    ax1[1].plot(pressure[0],pressureMisfit,'kx')
    ax1[1].plot(pressure[0],np.zeros(len(pressureMisfit)),'r--')
    ax1[0].set_xlim([0, t[-1]])
    ax1[1].set_xlim([0, t[-1]])
    ax1[0].set_ylabel('Pressure [Pa]')
    ax1[0].set_xlabel('time [days]')
    ax1[0].set_title('Comparison of model to observed pressure value')
    ax1[0].legend(loc='upper right',prop={'size': 7})
    ax1[1].set_ylabel('pressure misfit [Pa]')
    ax1[1].set_xlabel('time [days]')
    ax1[1].set_title('Best fit LMP model')

    f2,ax2 = plt.subplots(1,2)
    ax2[0].plot(tsol,T,'k--')
    ax2[0].plot(temp[0],temp[1],'r.',label='data')
    ax2[0].plot(t, np.ones(len(t)) * 240, 'g-', label = 'Toxic contaminant dissociation temperature')
    ax2[1].plot(temp[0],tempMisfit,'kx')
    ax2[1].plot(temp[0],temp[0]*0,'r--')
    ax2[0].set_xlim([0, t[-1]])
    ax2[1].set_xlim([0, t[-1]])
    ax2[0].set_ylabel('temperature [°C]')
    ax2[0].set_xlabel('time [days]')
    ax2[0].set_title('Comparison of model to observed \n temperature value')
    ax2[0].legend(loc='lower left',prop={'size': 7})
    ax2[1].set_ylabel('temperature misfit [°C]')
    ax2[1].set_xlabel('time [days]')
    ax2[1].set_title('Best fit LMP model')

    plt.show()


















