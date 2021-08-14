import pandas as pd
from os import sep
from functools import reduce


def odePressure(t, P, a, b, q, P0):
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

    dPdt = -a*q - b*(P - P0)
    return dPdt

def odeTemp(t, T, P, T0, P0, Tsteam, Tdash, a, b, c):
    ''' Return the derivative dT/dt at a time, t for given parameters.
        dT/dt = a(Tsteam - T) - b(P - P0)(Tdash - T) - c(T - T0)

        Parameters:
        -----------
        t : float
            Independent variable.
        T : float
            Dependent varaible.
        P : float
            Pressure value.
        T0 : float
            initial Temperature
        P0 : float
            initial Pressure
        Tdash : float
            function returning value for T'(t)
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
    Tprime = Tdash(t, P, T, P0, T0)
    dTdt = a*(Tsteam - T) - b*(P - P0)*(Tprime - T0) - c*(T - T0)

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

def objective():

    pass

if __name__ == "__main__":
    data = loadGivenData()


