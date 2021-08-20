import pandas as pd
import numpy as np
from os import sep
from functools import reduce
from scipy.optimize import curve_fit

Tsteam = 260 #degrees Celsius

def odePressure(X, a, b, P0):
    ''' Return the derivative dP/dt at a time, t for given parameters.

        Parameters:
        -----------
        X : float
            independent variables of form : (t, P, T, qw, qo, qs)
            where:
                t is time.
                P is pressure.
                T is temperature.
                qw is flow of water.
                qo is flow of oil.
                qs is flow of steam.
        a : float
            Extraction/injection parameter.
        b : float
            Recharge strength parameter.
        P0 : float
            Initial Pressure value.

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
    t, P, T, qw, qo, qs = X

    q = qw + qo - qs
    dPdt = -a*q - b*(P - P0)

    return dPdt


def modelPressure(X, a, b, P0):

    t, *unused = X
    tp, P = solve_ode(odePressure,t0=t[0], t1=t[-1], dt=0.5, x0=P0, X=X, pars=[a, b, P0])

    return P

def fitPressure():


    data = loadGivenData()

    t = data.index
    P = data[' pressure (kPa)']
    T = data['temperature (degC)']
    qo = data[' produced oil (m^3/day)']
    qs = data[' injected steam (tonnes/day)']
    qw = data[' produced water (m^3/day)']

    X = (t, P, T, qw, qo, qs)

    params,_ = curve_fit(lambda X, a, b: modelPressure(X, a, b, P.iloc[0]), X, P)


    pass

def readPressureData():
    data = pd.read_csv("data" + sep + "tr_p.txt", delimiter=',')

    return data


def odeTemp(X, a, b, P0, T0, M0, Tdash, bT):
    ''' Return the derivative dT/dt at a time, t for given parameters.

        Parameters:
        -----------
        X : float
            independent variables of form : (t, P, T, qw, qo, qs)
            where:
                t is time.
                P is pressure.
                T is temperature.
                qw is water rate.
                qo is oil rate.
                qs is steam rate.
        a : float
            Source/sink strength parameter.
        b : float
            Recharge strength parameter.
        T0 : float
            initial Temperature
        P0 : float
            initial Pressure
        M0 : float
            initial Mass
        Tdash : float
            function returning value for T'(t)
        bT : float
            conduction parameter

        Returns:
        --------
        dTdt : float
            Derivative of dependent variable with respect to independent variable.

        Notes:
        ------
        None

        Examples:
        ---------
        >>> 

    '''
    t, P, T, qw, qo, qs = X
    Tprime = Tdash(t, P, P0)

    dTdt = qs/M0 * (Tsteam - T) - b/(a*M0) * (P - P0) * (Tprime - T0) - bT*(T - T0)
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
            Returns the required value for temperature depending on directin of flow.
    '''
    # if (P > P0):
    #     return T
    # else:
    #     return T0
    
    Tprime = np.where(P > P0, T, T0)

    return Tprime

def improved_euler_step(f, t, x, dt, X, pars=[]):
    """Performs an impr0ved euler step.

	Parameters
	----------
		f : function name
			Derivative function.
		t : float
			Time.
		x : float
			Population.
		dt : float
			Step size.
		pars : list
			Contains additional parameters for derivative function.
		
	Returns
	-------
		x_plus1 : float
			improved euler step result.
	"""

    t, P, T, qw, qo, qs = X
    
    dxdt = f(X, *pars)
    dxdt_plus1 = f(t + dt, x + dt*dxdt, *pars)

    x_plus1 = x + dt*0.5*(dxdt + dxdt_plus1)

    return x_plus1



def solve_ode(f, t0, t1, dt, x0, X, pars=[]):
    ''' Solve an ODE numerically.

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

        Returns:
        --------
        t : array-like
            Independent variable solution vector.
        x : array-like
            Dependent variable solution vector.

        Notes:
        ------
        ODE should be solved using the Improved Euler Method. 

        Function q(t) should be hard coded within this method. Create duplicates of 
        solve_ode for models with different q(t).

        Assume that ODE function f takes the following inputs, in order:
            1. independent variable
            2. dependent variable
            3. forcing term, q
            4. all other parameters
    '''
    
    n = int(np.ceil(t1 - t0)/dt)        # compute number of Euler steps to take
    t = t0 + np.arange(n+1)*dt          # t array                               
    x = 0.*t                            # array to store solution               
    x[0] = x0                           # set initial value

    t, P, T, qw, qo, qs = X

    for i in range(n):
        x[i+1] = improved_euler_step(f, t[i], x[i], dt, X, pars)

    return t, x

def loadGivenData():
    oil = pd.read_csv("data" + sep + "tr_oil.txt")
    pressure = pd.read_csv("data" + sep + "tr_p.txt")
    steam = pd.read_csv("data" + sep + "tr_steam.txt")
    temp = pd.read_csv("data" + sep + "tr_T.txt")
    water = pd.read_csv("data" + sep + "tr_water.txt")

    dataArray = [oil, pressure, steam, temp, water]
    dataArray = [df.set_index('days') for df in dataArray]

    data = reduce(lambda left, right: pd.merge(left, right, on = ['days'], how = 'outer'), dataArray).sort_index() 

    data = data.interpolate(method='index')

    data = data.dropna()
    

    return data

if __name__ == "__main__":
    data = loadGivenData()
    fitPressure()


