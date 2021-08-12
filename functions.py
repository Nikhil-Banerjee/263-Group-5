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

def odeTemp(t, T, a, b, P, qsteam, T0, P0, M0, Tsteam, Tdash, bT):
    ''' Return the derivative dT/dt at a time, t for given parameters.

        Parameters:
        -----------
        t : float
            Independent variable.
        T : float
            Dependent varaible.
        a : float
            Source/sink strength parameter.
        b : float
            Recharge strength parameter.
        P : float
            Pressure value.
        qsteam : float
            steam rate
        T0 : float
            initial Temperature
        P0 : float
            initial Pressure
        M0 : float
            initial Mass
        Tsteam : float
            steam Temperature?
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
    Tprime = Tdash(t, P, P0)
    dTdt = qsteam/M0 * (Tsteam - T) - b/(a*M0) * (P - P0) * (Tprime - T0) - bT*(T - T0)

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

    data = reduce(lambda left, right: pd.merge(left, right, on = ['days'], how = 'outer'), dataArray)

    return data

# if __name__ == "__main__":
#     data = loadGivenData()


