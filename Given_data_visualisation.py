import pandas as pd
import os
from matplotlib import pyplot as plt
import datetime
from functools import reduce

# Reading in pilot study exp2 and exp3 files.
exp2Oil = pd.read_csv("pilot_project_data_2012" + os.sep + "exp2_oil.csv")
exp2Pressure = pd.read_csv("pilot_project_data_2012" + os.sep + "exp2_Pressure.csv")
exp2Steam = pd.read_csv("pilot_project_data_2012" + os.sep + "exp2_Steam.csv")
exp2Temp = pd.read_csv("pilot_project_data_2012" + os.sep + "exp2_Temp.csv")
exp2Water = pd.read_csv("pilot_project_data_2012" + os.sep + "exp2_Water.csv")

exp3Oil = pd.read_csv("pilot_project_data_2012" + os.sep + "exp3_oil.csv")
exp3Pressure = pd.read_csv("pilot_project_data_2012" + os.sep + "exp3_Pressure.csv")
exp3Steam = pd.read_csv("pilot_project_data_2012" + os.sep + "exp3_Steam.csv")
exp3Temp = pd.read_csv("pilot_project_data_2012" + os.sep + "exp3_Temp.csv")
exp3Water = pd.read_csv("pilot_project_data_2012" + os.sep + "exp3_Water.csv")


jan2012 = datetime.datetime(2012, 1, 1)
# Joining exp2 dfs
dfs2 = [exp2Oil, exp2Pressure, exp2Steam, exp2Temp, exp2Water]

dfs2 = [df.set_index('days since jan 2012') for df in dfs2]
dfs2merged = reduce(lambda left,right: pd.merge(left, right, on = ['days since jan 2012'], how = 'outer'), dfs2)

dfs2merged.insert(1, "date", value=0)

for i, row in dfs2merged.iterrows():
    dfs2merged.at[i, 'date'] = jan2012 + datetime.timedelta(days = i)


jun2012 = datetime.datetime(2012, 6, 1)
# Joining exp3 dfs
dfs3 = [exp3Oil, exp3Pressure, exp3Steam, exp3Temp, exp3Water]

dfs3 = [df.set_index('days since jun 2012') for df in dfs3]
dfs3merged = reduce(lambda left,right: pd.merge(left, right, on = ['days since jun 2012'], how = 'outer'), dfs3)

dfs3merged.insert(1, "date", value=0)

for i, row in dfs3merged.iterrows():
    dfs3merged.at[i, 'date'] = jun2012 + datetime.timedelta(days = i)


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
f2.suptitle('Experiment 2 Data')

ax2[0].plot(dfs2merged['date'], dfs2merged['steam rate (t/d)'], marker = 'o', linestyle = 'none', color = steamCol ,fillstyle = 'none' ,label = 'Steam Rate (t/d)')
ax2[0].set_ylabel('Steam Rate (t/d)')
ax2[0].legend()

l2_1a = ax2[1].plot(dfs2merged['date'], dfs2merged['water rate (m^3/day)'], marker = 'x', linestyle = 'none', color = waterCol, fillstyle = 'none', label = 'Water Rate (m^3/day)')
ax2[1].set_ylabel("Water Rate (m^3/day)", color = waterCol)
ax2[1].tick_params(axis='y', colors = waterCol)
ax2[1].title.set_color(waterCol)

ax2twin1 = ax2[1].twinx()
l2_1b = ax2twin1.plot(dfs2merged['date'], dfs2merged['oil rate (m^3/day)'], marker = '^', linestyle = 'none', color = oilCol, fillstyle = 'none',label = 'oil rate (m^3/day)')
ax2twin1.set_ylabel("Oil Rate (m^3/day)", color = oilCol)
ax2twin1.tick_params(axis='y', colors = oilCol)
ax2twin1.title.set_color(oilCol)

l2_1 = l2_1a + l2_1b
lab2_1 = [l.get_label() for l in l2_1]
ax2[1].legend(l2_1, lab2_1)

l2_2a = ax2[2].plot(dfs2merged['date'], dfs2merged['pressure (kPa)'], color = pressureCol, label = 'pressure (kPa)')
ax2[2].set_ylabel("Pressure (kPa)", color = pressureCol)
ax2[2].tick_params(axis='y', colors = pressureCol)
ax2[2].title.set_color(pressureCol)

ax2twin2 = ax2[2].twinx()
l2_2b = ax2twin2.plot(dfs2merged['date'], dfs2merged['temperature (degC)'], color = tempCol, label = 'Temperature (°C)')
ax2twin2.set_ylabel("Temperature (°C)", color = tempCol)
ax2twin2.tick_params(axis='y', colors = tempCol)
ax2twin2.title.set_color(tempCol)

l2_2 = l2_2a + l2_2b
lab2_2 = [l.get_label() for l in l2_2]
ax2[2].legend(l2_2, lab2_2)

[ax.grid() for ax in ax2] 


# exp3 plots
f3 = plt.figure()
g3 = f3.add_gridspec(3, hspace=0.1)
ax3 = g3.subplots(sharex=True)
f3.suptitle('Experiment 3 Data')

ax3[0].plot(dfs3merged['date'], dfs3merged['steam rate (t/d)'], marker = 'o', linestyle = 'none', color = steamCol,fillstyle = 'none' ,label = 'Steam Rate (t/d)')
ax3[0].set_ylabel('Steam Rate (t/d)')
ax3[0].legend()


l3_1a = ax3[1].plot(dfs3merged['date'], dfs3merged['water rate (m^3/d)'], marker = 'x', linestyle = 'none', color = waterCol, fillstyle = 'none', label = 'Water Rate (m^3/day)')
ax3[1].set_ylabel("Water Rate (m^3/day)", color = waterCol)
ax3[1].tick_params(axis='y', colors = waterCol)
ax3[1].title.set_color(waterCol)

ax3twin1 = ax3[1].twinx()
l3_1b = ax3twin1.plot(dfs3merged['date'], dfs3merged['oil rate (m^3/d)'], marker = '^', linestyle = 'none', color = oilCol, fillstyle = 'none',label = 'oil rate (m^3/day)')
ax3twin1.set_ylabel("Oil Rate (m^3/day)", color = oilCol)
ax3twin1.tick_params(axis='y', colors = oilCol)
ax3twin1.title.set_color(oilCol)

l3_1 = l3_1a + l3_1b
lab3_1 = [l.get_label() for l in l3_1]
ax3[1].legend(l3_1, lab3_1)

l3_2a = ax3[2].plot(dfs3merged['date'], dfs3merged['pressure (kPa)'], color = pressureCol, label = 'pressure (kPa)')
ax3[2].set_ylabel("Pressure (kPa)", color = pressureCol)
ax3[2].tick_params(axis='y', colors = pressureCol)
ax3[2].title.set_color(pressureCol)

ax3twin2 = ax3[2].twinx()
l3_2b = ax3twin2.plot(dfs3merged['date'], dfs3merged['temperature (degC)'], color = tempCol, label = 'Temperature (°C)')
ax3twin2.set_ylabel("Temperature (°C)", color = tempCol)
ax3twin2.tick_params(axis='y', colors = tempCol)
ax3twin2.title.set_color(tempCol)

l3_2 = l3_2a + l3_2b
lab3_2 = [l.get_label() for l in l3_2]
ax3[2].legend(l3_2, lab3_2)

[ax.grid() for ax in ax3] 

plt.show()