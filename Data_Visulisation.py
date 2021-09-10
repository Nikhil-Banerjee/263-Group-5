import numpy as np
from os import sep
from matplotlib import pyplot as plt
from functools import reduce


oil = np.genfromtxt("data" + sep + "tr_oil.txt",delimiter=',',skip_header=1).T
pressure = np.genfromtxt("data" + sep + "tr_p.txt",delimiter=',',skip_header=1).T
steam = np.genfromtxt("data" + sep + "tr_steam.txt",delimiter=',',skip_header=1).T
temp = np.genfromtxt("data" + sep + "tr_T.txt",delimiter=',',skip_header=1).T
water = np.genfromtxt("data" + sep + "tr_water.txt",delimiter=',',skip_header=1).T

# Controls the colors of the plots:
steamCol = 'black'
waterCol = 'blue'
oilCol = '#EACE09'
pressureCol = 'green'
tempCol = 'red'

# exp2 plots
f2 = plt.figure()
g2 = f2.add_gridspec(3, hspace=0.1)
ax2 = g2.subplots(sharex=True)
f2.suptitle('Given Data')

ax2[0].plot(steam[0], steam[1], marker = 'o', linestyle = 'none', color = steamCol ,fillstyle = 'none' ,label = 'Steam Rate (t/d)')
ax2[0].set_ylabel('Steam Rate (t/d)')
ax2[0].legend()

l2_1a = ax2[1].plot(water[0], water[1], marker = 'x', linestyle = 'none', color = waterCol, fillstyle = 'none', label = 'Water Rate (m^3/day)')
ax2[1].set_ylabel("Water Rate (m^3/day)", color = waterCol)
ax2[1].tick_params(axis='y', colors = waterCol)
ax2[1].title.set_color(waterCol)

ax2twin1 = ax2[1].twinx()
l2_1b = ax2twin1.plot(oil[0], oil[1], marker = '^', linestyle = 'none', color = oilCol, fillstyle = 'none',label = 'oil rate (m^3/day)')
ax2twin1.set_ylabel("Oil Rate (m^3/day)", color = oilCol)
ax2twin1.tick_params(axis='y', colors = oilCol)
ax2twin1.title.set_color(oilCol)

l2_1 = l2_1a + l2_1b
lab2_1 = [l.get_label() for l in l2_1]
ax2[1].legend(l2_1, lab2_1)

l2_2a = ax2[2].plot(pressure[0], pressure[1], color = pressureCol, label = 'pressure (kPa)')
ax2[2].set_ylabel("Pressure (kPa)", color = pressureCol)
ax2[2].tick_params(axis='y', colors = pressureCol)
ax2[2].title.set_color(pressureCol)

ax2twin2 = ax2[2].twinx()
l2_2b = ax2twin2.plot(temp[0], temp[1], color = tempCol, label = 'Temperature (°C)')
ax2twin2.set_ylabel("Temperature (°C)", color = tempCol)
ax2twin2.tick_params(axis='y', colors = tempCol)
ax2twin2.title.set_color(tempCol)

l2_2 = l2_2a + l2_2b
lab2_2 = [l.get_label() for l in l2_2]
ax2[2].legend(l2_2, lab2_2)

[ax.grid() for ax in ax2] 

plt.show()





