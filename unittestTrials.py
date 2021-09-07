import numpy as np
from os import sep
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from benchmark import*
from functionsFinal import*

Q = Qterms()

trial1 = odePressure(0, 0, Q.giveQ, 0, 0, 0)
print(trial1)

trial2 = odePressure(1, 1, Q.giveQ, 1, 1, 1)
print(trial2)

trial3 = odePressure(5, 8, Q.giveQ, 9, 7, 4)
print(trial3)

trial4 = odePressure(-1, -1, Q.giveQ, -1, -1, -1)
print(trial4)

trial5 = odePressure("manas", 0, Q.giveQ, 0, 0, 0)
print(trial5)
