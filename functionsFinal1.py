import numpy as np
from os import sep
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from benchmark import*

Tsteam = 260

def odePressure(t, P, q, a, b, P0):
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
        q(t) = qw(t) - qs(t) + qo(t)
        where:
        qw is flow rate of water
        qs is flow rate of steam
        qo is flow rate of oil
    P0 : float
        Initial Pressure value
  
    Returns:
    --------
    dPdt : float
           Derivative of Pressure with respect to time.
    '''
    dPdt = -a*q(t) - b*(P - P0)
    return dPdt

def odeTemp(t, T, P, qs, a, b, c, M, P0, T0):
    '''Returns dT/dt

    Parameters:
    -----------
    t : float
        Independent time variable.
    P : float
        Dependent Pressure variable.
    T : float
        Dependent Temperature variable.
    qs : function
        Flow rate of steam - qs(t)
    a : float
        extraction/injection parameter
    b : float
        recharge parameter
    c: float
        conduction parameter
    M : float
        Mass of the system.
    P0 : float
        initial Pressure
    T0 : float
        initial Temperature
    
    Returns:
    --------
    dTdt : float
        Derivative of Temperature with respect to time.
    '''
    dTdt = qs(t)/M*(Tsteam - T) - b/(a*M)*(P - P0)*(Tprime(t, P, T, P0, T0) - T) - c*(T - T0)
    return dTdt

def Tprime(t, P, T, P0, T0):
    # Helper function for odeTemp.
    if (P > P0):
        return T
    else:  
        return T0

def solvePressure(t, dt, P0, q, pars):
    x1=np.zeros(len(t))
    x1[0]=P0
    x=np.zeros(len(t))
    x[0]=P0

    # improved euler method
    for i in range(len(t)-1):
        k1=odePressure(t[i],x1[i],q,*pars)
        x1[i+1]=k1*dt+x1[i]
        k2=odePressure(t[i+1],x1[i+1],q,*pars)
        x[i+1]=dt*(k1+k2)*0.5+x1[i]
    return t, x  

def solveTemperature(t, dt, P, qs, a, b, c, M, P0, T0):

    x1=np.zeros(len(t))
    x1[0]=T0
    x=np.zeros(len(t))
    x[0]=T0
    # improved euler method
    for i in range(len(t)-1):
        k1=odeTemp(t[i], x1[i], P[i], qs, a, b, c, M, P0, T0)
        x1[i+1]=k1*dt+x1[i]
        k2=odeTemp(t[i+1], x1[i+1], P[i+1], qs, a, b, c, M, P0, T0)
        x[i+1]=dt*(k1+k2)*0.5+x1[i]

    return t,x

def fitPressure(t, q, a, b, P0):
    ts, P = solvePressure(t, t[1]-t[0], P0, q, [a,b,P0])

    return P

def fitTemp(t, q, qs, a, b, c, M, P0, T0):
    P = fitPressure(t, q, a, b, P0)

    tsol, T = solveTemperature(t, t[1]-t[0], P, qs, a, b, c, M, P0, T0)
    
    return T

class Qterms:
    def __init__(self, forcInject = None):
        # Initializes the q(t) and qs(t) function object by loading all required data.
        oil, pressure, steam, temp, water=loadData()  
        self.oil = oil
        self.pressure = pressure
        self.steam = steam
        self.temp = temp
        self.water = water

        # Interpolating functions for the given values.
        self.interpWater = interp1d(water[0], water[1])
        self.interpOil = interp1d(oil[0], oil[1])
        self.interpSteam = interp1d(steam[0], steam[1])

        # Storing the time values for injection end and projection end of forecasts.
        self.forcInjecEnd = temp[0][-1] + 60
        self.forcProducEnd = self.forcInjecEnd + 90

        # Assigns value to be used for qs during injection period.
        if forcInject != None:
            self.forc_s = forcInject
        else:
            self.forc_s = self.steam[1][-1]
    
    def giveQ(self, t):
        ''' Returns q(t) value for non-forecasts and forecasts.

            Parameters:
            -----------
            t : float
                independent time variable

            Returns:
            --------
            q : float
                q(t) value
                - qwater - qsteam + qoil
        '''

        # Finds required qwater value for required time.
        if (t < self.water[0][0]):
            w = self.water[1][0]
        elif ((t >= self.water[0][0]) and (t <= self.water[0][-1])):
            w = self.interpWater(t)
        elif ((t > self.water[0][-1]) and (t <= self.forcInjecEnd)):
            w = 0
        elif ((t > self.forcInjecEnd)) and (t <= self.forcProducEnd):
            w = 80.637/2
        else:
            ValueError('Tried to access Q outside of allowed time values.')
    
        # Finds required qoil value for required time.
        if (t < self.oil[0][0]):
            o = self.oil[1][0]
        elif ((t >= self.oil[0][0]) and (t <= self.oil[0][-1])):
            o = self.interpOil(t)
        elif ((t > self.oil[0][-1]) and (t <= self.forcInjecEnd)):
            o = 0
        elif ((t > self.forcInjecEnd)) and (t <= self.forcProducEnd):
            o = 80.637/2
        else:
            ValueError('Tried to access Q outside of allowed time values.')
        
        # Finds required qsteam value for required time.
        s = self.giveQs(t)

        q = w-s+o
        return q
    
    def giveQs(self, t):
        ''' Returns qsteam(t) value for non-forecasts and forecasts.

            Parameters:
            -----------
            t : float
                independent time variable

            Returns:
            --------
            s : float
                Flow rate of steam
        '''
        
        # Returns the required qs value for the timeframe
        if (t < self.steam[0][0]):
            # Before first recorded qs.
            s = self.steam[1][0]
        elif((t >= self.steam[0][0]) and (t <= self.steam[0][-1])):
            # Within given qs values.
            s = self.interpSteam(t)
        elif ((t > self.steam[0][-1]) and (t <= self.forcInjecEnd)):
            # For forecasted injection period.
            s = self.forc_s
        elif ((t > self.forcInjecEnd)) and (t <= self.forcProducEnd):
            # For forecasted production period.
            s = 0
        else:
            ValueError('Tried to access Qs outside of allowed time values.')

        return s

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

if __name__ == "__main__":
    oil, pressure, steam, temp, water=loadData()

    t1=np.linspace(0,217,1000)

    # first pressure model initial guesses:
    a = 0.2
    b = 0.05

    Q = Qterms() # Creates qterm instance for non forecasting fitting purposes.

    # Modifies function so that it leaves only a,b as parameters apart from t.
    callibP = lambda t, a, b: fitPressure(t, Q.giveQ, a, b, pressure[1][0])

    parsFoundP, pcov = curve_fit(callibP, pressure[0], pressure[1], [a,b])
    print(parsFoundP)

    tsolP, P = solvePressure(t1, t1[1]-t1[0], pressure[1][0], Q.giveQ, [*parsFoundP,pressure[1][0]])

    pressureMisfit = pressure[1] - np.interp(pressure[0], tsolP, P)

    f9,ax9 = plt.subplots(1,2)
    ax9[0].plot(tsolP,P,'k--')
    ax9[0].plot(pressure[0],pressure[1],'r.',label='data')
    ax9[1].plot(pressure[0],pressureMisfit,'kx')
    ax9[1].plot(pressure[0],np.zeros(len(pressureMisfit)),'r--')
    ax9[0].plot(t1,P,'k--',label='a = {:3f}\nb = {:3f}'.format(parsFoundP[0],parsFoundP[1]))
    ax9[0].set_xlim([0, tsolP[-1]])
    ax9[1].set_xlim([0, tsolP[-1]])
    ax9[0].set_ylabel('Pressure [Pa]')
    ax9[0].set_xlabel('time [days]')
    ax9[0].set_title('Comparison of model to observed pressure value')
    ax9[0].legend(loc='upper right',prop={'size': 7})
    ax9[1].set_ylabel('pressure misfit [Pa]')
    ax9[1].set_xlabel('time [days]')
    ax9[1].set_title('Best fit LMP model')

    # first Temperature model initial guess
    c = 0.1
    M = 4000

    callibT = lambda t,c,M: fitTemp(t, Q.giveQ, Q.giveQs,\
         parsFoundP[0], parsFoundP[1], c, M, pressure[1][0], temp[1][0])

    parsFoundT, tcov = curve_fit(callibT, temp[0], temp[1], [c, M])
    print(parsFoundT)

    tsolT, T = solveTemperature(t1, t1[1]-t1[0], P, Q.giveQs, parsFoundP[0],\
        parsFoundP[1], parsFoundT[0], parsFoundT[1], pressure[1][0], temp[1][0])

    tempMisfit = temp[1] - np.interp(temp[0], tsolT, T)

    f8,ax8 = plt.subplots(1,2)
    ax8[0].plot(t1,T,'k--',label='c = {:3f}\nM = {:3f}'.format(parsFoundT[0],parsFoundT[1]))
    ax8[0].plot(temp[0],temp[1],'r.',label='data')
    ax8[0].plot(t1, np.ones(len(t1)) * 240, 'g-', label = 'Toxic contaminant\ndissociation temperature')
    ax8[1].plot(temp[0],tempMisfit,'kx')
    ax8[1].plot(temp[0],temp[0]*0,'r--')
    ax8[0].set_xlim([0, t1[-1]])
    ax8[1].set_xlim([0, t1[-1]])
    ax8[0].set_ylabel('temperature [°C]')
    ax8[0].set_xlabel('time [days]')
    ax8[0].set_title('Comparison of model to observed \n temperature value')
    ax8[0].legend(loc='lower left',prop={'size': 7})
    ax8[1].set_ylabel('temperature misfit [°C]')
    ax8[1].set_xlabel('time [days]')
    ax8[1].set_title('Best fit LMP model')
    
    # Improved Pressure model initial guesses:
    a = 0.2
    b = 0.05
    P0 = pressure[1][0]

    Q = Qterms() # Creates qterm instance for non forecasting fitting purposes.

    # Modifies function so that it leaves only a,b,P0 as parameters apart from t.
    callibP = lambda t, a, b, P0: fitPressure(t, Q.giveQ, a, b, P0)

    parsFoundP, pcov = curve_fit(callibP, pressure[0], pressure[1], [a,b,P0])
    print(parsFoundP)

    tsolP, P = solvePressure(t1, t1[1]-t1[0], parsFoundP[2], Q.giveQ, parsFoundP)

    pressureMisfit = pressure[1] - np.interp(pressure[0], tsolP, P)

    f1,ax1 = plt.subplots(1,2)
    ax1[0].plot(tsolP,P,'k--')
    ax1[0].plot(pressure[0],pressure[1],'r.',label='data')
    ax1[1].plot(pressure[0],pressureMisfit,'kx')
    ax1[1].plot(pressure[0],np.zeros(len(pressureMisfit)),'r--')
    ax1[0].plot(t1,P,'k--',label='a = {:3f}\nb = {:3f}\nP0 = {:3f}'.format(parsFoundP[0],parsFoundP[1],parsFoundP[2]))
    ax1[0].set_xlim([0, tsolP[-1]])
    ax1[1].set_xlim([0, tsolP[-1]])
    ax1[0].set_ylabel('Pressure [Pa]')
    ax1[0].set_xlabel('time [days]')
    ax1[0].set_title('Comparison of model to observed pressure value')
    ax1[0].legend(loc='upper right',prop={'size': 7})
    ax1[1].set_ylabel('pressure misfit [Pa]')
    ax1[1].set_xlabel('time [days]')
    ax1[1].set_title('Best fit LMP model')

    # improved Temperature model initial guesses
    c = 0.1
    M = 4000
    T0 = temp[1][0]

    callibT = lambda t,c,M,T0: fitTemp(t, Q.giveQ, Q.giveQs,\
         parsFoundP[0], parsFoundP[1], c, M, parsFoundP[2], T0)

    parsFoundT, tcov = curve_fit(callibT, temp[0], temp[1], [c, M, T0])
    print(parsFoundT)

    tsolT, T = solveTemperature(t1, t1[1]-t1[0], P, Q.giveQs, parsFoundP[0],\
        parsFoundP[1], parsFoundT[0], parsFoundT[1], parsFoundP[2], parsFoundT[2])

    tempMisfit = temp[1] - np.interp(temp[0], tsolT, T)

    f2,ax2 = plt.subplots(1,2)
    ax2[0].plot(t1,T,'k--',label='c = {:3f}\nM = {:3f}\nT0 = {:3f}'.format(parsFoundT[0],parsFoundT[1],parsFoundT[2]))
    ax2[0].plot(temp[0],temp[1],'r.',label='data')
    ax2[0].plot(t1, np.ones(len(t1)) * 240, 'g-', label = 'Toxic contaminant\ndissociation temperature')
    ax2[1].plot(temp[0],tempMisfit,'kx')
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

    # Benchmarking Code HERE:
    benchmark()



    # Forecasts:
    tf = np.linspace(0, 370, 500)
    # Forecast 1
    # Tood Energy proposal of steam injection of 1000 tonnes per day 60 days, followed by 90 day production periods.
    Qf1 = Qterms(1000)
    t, FP1 = solvePressure(tf, tf[1]-tf[0], parsFoundP[2], Qf1.giveQ, parsFoundP)
    t, FT1 = solveTemperature(tf, tf[1]-tf[0], FP1, Qf1.giveQs, parsFoundP[0],\
        parsFoundP[1], parsFoundT[0], parsFoundT[1], parsFoundP[2], parsFoundT[2])
    
    # Forecast 2
    # No steam injection
    Qf2 = Qterms(0)
    t, FP2 = solvePressure(tf, tf[1]-tf[0], parsFoundP[2], Qf2.giveQ, parsFoundP)
    t, FT2 = solveTemperature(tf, tf[1]-tf[0], FP1, Qf2.giveQs, parsFoundP[0],\
        parsFoundP[1], parsFoundT[0], parsFoundT[1], parsFoundP[2], parsFoundT[2])
    
    # Forecast 3
    # Current steam injection of 460 tonnes per day for 60 days, followed by 90 day production periods.
    Qf3 = Qterms(460)
    t, FP3 = solvePressure(tf, tf[1]-tf[0], parsFoundP[2], Qf3.giveQ, parsFoundP)
    t, FT3 = solveTemperature(tf, tf[1]-tf[0], FP1, Qf3.giveQs, parsFoundP[0],\
        parsFoundP[1], parsFoundT[0], parsFoundT[1], parsFoundP[2], parsFoundT[2])
    
    # Forecast 4
    # steam injection of 200 tonnes per day 60 days, followed by 90 day production periods.
    Qf4 = Qterms(200)
    t, FP4 = solvePressure(tf, tf[1]-tf[0], parsFoundP[2], Qf4.giveQ, parsFoundP)
    t, FT4 = solveTemperature(tf, tf[1]-tf[0], FP1, Qf4.giveQs, parsFoundP[0],\
        parsFoundP[1], parsFoundT[0], parsFoundT[1], parsFoundP[2], parsFoundT[2])

    startForc = np.argmax(tf>=t1[-1])

    # Forecast Temperatures plot
    tOverall = np.arange(t1[0],tf[-1])
    f4, ax4 = plt.subplots(1,1)
    ax4.plot(t1,T,'b-',label='Model')
    ax4.plot(temp[0],temp[1],'ko',label='data')
    ax4.plot(tf[startForc:], FT4[startForc:], color = '#8B008B', ls = '-', label = 'Steam injection = 200 t/d')
    ax4.plot(tf[startForc:], FT1[startForc:], 'y-', label = 'Todd Energy proposed\nsteam injection = 1000 t/d')
    ax4.plot(tf[startForc:], FT3[startForc:], color = '#00FFFF', ls = '-', label = 'Current steam injection = 460 t/d')
    ax4.plot(tf[startForc:], FT2[startForc:], 'g-', label = 'Steam injection = 0 t/d')
    ax4.plot(tOverall,np.ones(len(tOverall)) * 240,'r-', label =  'Toxic contaminant\ndissociation temperature')
    ax4.legend(loc='upper right',prop={'size': 7},bbox_to_anchor=(1.1, 0.95))
    ax4.text(x = 10, y = 124, s = 'All forecasted injection phases are 60 days followed by 90 day production periods.', bbox = dict(facecolor='none', edgecolor='black', pad=5.0))
    ax4.set_xlim([t1[0], tf[-1]])
    ax4.set_ylim([120, 245])
    ax4.set_title('Temperature Forecasting')
    ax4.set_xlabel('time (days)')
    ax4.set_ylabel('Temperature (°C)')

    # Forecast Pressure plot
    f5, ax5 = plt.subplots(1,1)
    ax5.plot(t1, P, 'b-', label = 'Model')
    ax5.plot(pressure[0], pressure[1], 'ko', label = 'data')
    ax5.plot(tf[startForc:], FP4[startForc:], color = '#8B008B', ls = '-', label = 'Steam injection = 200 t/d')
    ax5.plot(tf[startForc:], FP1[startForc:], 'y-', label = 'Todd Energy proposed steam injection = 1000 t/d')
    ax5.plot(tf[startForc:], FP3[startForc:], color = '#00FFFF', ls = '-', label = 'Current steam injection = 460 t/d')
    ax5.plot(tf[startForc:], FP2[startForc:], 'g-', label = 'Steam injection = 0 t/d')
    ax5.legend()
    ax5.text(x = 10, y = 130, s = 'All forecasted injection phases are 60 days followed by 90 day production periods.', bbox = dict(facecolor='none', edgecolor='black', pad=5.0))
    ax5.set_xlim([t1[0], tf[-1]])
    ax5.set_title('Pressure Forecasting')
    ax5.set_xlabel('time (days)')
    ax5.set_ylabel('Pressure (Pa)')

    # uncertainty for pressure plot
    fig,ax = plt.subplots(1,1)
    ax.plot(t1, P, 'b-', label = 'Model')
    ax.plot(pressure[0], pressure[1], 'ko', label = 'data')
    ax.plot(tf[startForc:], FP4[startForc:], color = '#8B008B', ls = '-', label = 'Steam injection = 200 t/d')
    ax.plot(tf[startForc:], FP1[startForc:], 'y-', label = 'Todd Energy proposed steam injection = 1000 t/d')
    ax.plot(tf[startForc:], FP3[startForc:], color = '#00FFFF', ls = '-', label = 'Current steam injection = 460 t/d')
    ax.plot(tf[startForc:], FP2[startForc:], 'g-', label = 'Steam injection = 0 t/d')
    ax.set_title('Pressure')
    ps = np.random.multivariate_normal(parsFoundP, pcov, 100)   # samples from posterior
    for pi in ps:
        tsolP, P = solvePressure(t1, t1[1]-t1[0], pi[2], Q.giveQ, pi)
        # Forecast 1
        # Tood Energy proposal of steam injection of 1000 tonnes per day 60 days, followed by 90 day production periods.
        Qf1 = Qterms(1000)
        t, FP1 = solvePressure(tf, tf[1]-tf[0], pi[2], Qf1.giveQ, pi)      
        # Forecast 2
        # No steam injection
        Qf2 = Qterms(0)
        t, FP2 = solvePressure(tf, tf[1]-tf[0], pi[2], Qf2.giveQ, pi)        
        # Forecast 3
        # Current steam injection of 460 tonnes per day for 60 days, followed by 90 day production periods.
        Qf3 = Qterms(460)
        t, FP3 = solvePressure(tf, tf[1]-tf[0], pi[2], Qf3.giveQ, pi) 
        # Forecast 4
        # steam injection of 200 tonnes per day 60 days, followed by 90 day production periods.
        Qf4 = Qterms(200)
        t, FP4 = solvePressure(tf, tf[1]-tf[0], pi[2], Qf4.giveQ, pi)
        ax.plot(tsolP, P, 'b-', alpha=0.2, lw=0.5)
        ax.plot(tf[startForc:], FP4[startForc:], color = '#8B008B', alpha=0.2, lw=0.5)
        ax.plot(tf[startForc:], FP1[startForc:], 'y-', alpha=0.2, lw=0.5)
        ax.plot(tf[startForc:], FP3[startForc:], color = '#00FFFF', alpha=0.2, lw=0.5)
        ax.plot(tf[startForc:], FP2[startForc:], 'g-', alpha=0.2, lw=0.5)
    ax.legend()  



    # uncertainity for temperature plot
    fig,ax6 = plt.subplots(1,1)
    ax6.plot(t1,T,'b-',label='Model')
    ax6.plot(temp[0],temp[1],'ko',label='data')
    ax6.plot(tf[startForc:], FT4[startForc:], color = '#8B008B', ls = '-', label = 'Steam injection = 200 t/d')
    ax6.plot(tf[startForc:], FT1[startForc:], 'y-', label = 'Todd Energy proposed\nsteam injection = 1000 t/d')
    ax6.plot(tf[startForc:], FT3[startForc:], color = '#00FFFF', ls = '-', label = 'Current steam injection = 460 t/d')
    ax6.plot(tf[startForc:], FT2[startForc:], 'g-', label = 'Steam injection = 0 t/d')
    ax6.plot(tOverall,np.ones(len(tOverall)) * 240,'r-', label =  'Toxic contaminant\ndissociation temperature')
    ax6.set_title('Temperature')
    ts = np.random.multivariate_normal(parsFoundT, tcov, 100)   # samples from posterior
    for pi in ts:
        # Forecast 1
        # Tood Energy proposal of steam injection of 1000 tonnes per day 60 days, followed by 90 day production periods.
        Qf1 = Qterms(1000)
        t, FP1 = solvePressure(tf, tf[1]-tf[0], parsFoundP[2], Qf1.giveQ, parsFoundP)
        t, FT1 = solveTemperature(tf, tf[1]-tf[0], FP1, Qf1.giveQs, parsFoundP[0],\
            parsFoundP[1], pi[0], pi[1], parsFoundP[2], pi[2])
        
        # Forecast 2
        # No steam injection
        Qf2 = Qterms(0)
        t, FP2 = solvePressure(tf, tf[1]-tf[0], parsFoundP[2], Qf2.giveQ, parsFoundP)
        t, FT2 = solveTemperature(tf, tf[1]-tf[0], FP1, Qf2.giveQs, parsFoundP[0],\
            parsFoundP[1], pi[0], pi[1], parsFoundP[2], pi[2])
        
        # Forecast 3
        # Current steam injection of 460 tonnes per day for 60 days, followed by 90 day production periods.
        Qf3 = Qterms(460)
        t, FP3 = solvePressure(tf, tf[1]-tf[0], parsFoundP[2], Qf3.giveQ, parsFoundP)
        t, FT3 = solveTemperature(tf, tf[1]-tf[0], FP1, Qf3.giveQs, parsFoundP[0],\
            parsFoundP[1], pi[0], pi[1], parsFoundP[2], pi[2])
        
        # Forecast 4
        # steam injection of 200 tonnes per day 60 days, followed by 90 day production periods.
        Qf4 = Qterms(200)
        t, FP4 = solvePressure(tf, tf[1]-tf[0], parsFoundP[2], Qf4.giveQ, parsFoundP)
        t, FT4 = solveTemperature(tf, tf[1]-tf[0], FP1, Qf4.giveQs, parsFoundP[0],\
            parsFoundP[1], pi[0], pi[1], parsFoundP[2], pi[2])    
        ax6.plot(tsolT, T, 'b-', alpha=0.2, lw=0.5)
        ax6.plot(tf[startForc:], FT4[startForc:], color = '#8B008B', alpha=0.2, lw=0.5)
        ax6.plot(tf[startForc:], FT1[startForc:], 'y-', alpha=0.2, lw=0.5)
        ax6.plot(tf[startForc:], FT3[startForc:], color = '#00FFFF', alpha=0.2, lw=0.5)
        ax6.plot(tf[startForc:], FT2[startForc:], 'g-', alpha=0.2, lw=0.5)
        tsolT, T = solveTemperature(t1, t1[1]-t1[0], P, Q.giveQs, parsFoundP[0],\
        parsFoundP[1], pi[0], pi[1], parsFoundP[2], pi[2])
        ax6.plot(t1,T,'b-',alpha=0.2,lw=0.5)
        ax6.legend(loc='upper right',prop={'size': 7},bbox_to_anchor=(1.1, 0.88))  
        ax6.set_xlim([0,370])

    plt.show()































