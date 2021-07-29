import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import datetime
from functools import reduce
# from glob import glob

# exp2Path = glob("pilot_project_data_2012" + os.sep + "exp2*.csv")
# exp3Path = glob("pilot_project_data_2012" + os.sep + "exp3*.csv")

# exp2Data = []
# exp3Data = []

# for file in exp2Path:
#     exp2Data.append(pd.read_csv(file))

# for file in exp3Path
#     exp3Data.append(pd.read(file))

exp2Oil = pd.read_csv("pilot_project_data_2012" + os.sep + "exp2_oil.csv")
exp2Pressure = pd.read_csv("pilot_project_data_2012" + os.sep + "exp2_Pressure.csv")
exp2Steam = pd.read_csv("pilot_project_data_2012" + os.sep + "exp2_Steam.csv")
exp2Temp = pd.read_csv("pilot_project_data_2012" + os.sep + "exp2_Temp.csv")
exp2Water = pd.read_csv("pilot_project_data_2012" + os.sep + "exp2_Water.csv")

exp3Oil = pd.read_csv("pilot_project_data_2012" + os.sep + "exp3_oil.csv", skiprows=[0,1,2,3,4])
exp3Pressure = pd.read_csv("pilot_project_data_2012" + os.sep + "exp3_Pressure.csv", skiprows=[0,1,2,3,4])
exp3Steam = pd.read_csv("pilot_project_data_2012" + os.sep + "exp3_Steam.csv", skiprows=[0,1,2,3,4])
exp3Temp = pd.read_csv("pilot_project_data_2012" + os.sep + "exp3_Temp.csv", skiprows=[0,1,2,3,4])
exp3Water = pd.read_csv("pilot_project_data_2012" + os.sep + "exp3_Water.csv", skiprows=[0,1,2,3,4])

# Reassigning indices as dates for exp2.
jan2012 = datetime.datetime(2012, 1, 31)

# Joining exp2 dfs
dfs2 = [exp2Oil, exp2Pressure, exp2Steam, exp2Temp, exp2Water]

dfs2 = [df.set_index('days since jan 2012') for df in dfs2]
dfs2merged = reduce(lambda left,right: pd.merge(left, right, on = ['days since jan 2012'], how = 'outer'), dfs2)

dfs2merged.insert(1, "date", value=0)

for i, row in dfs2merged.iterrows():
    dfs2merged.at[i, 'date'] = jan2012 + datetime.timedelta(days = i)

# Joining exp3 dfs
dfs3 = [exp3Oil, exp3Pressure, exp3Steam, exp3Temp, exp3Water]

dfs3 = [df.set_index('days since jun 2012') for df in dfs3]
dfs3merged = reduce(lambda left,right: pd.merge(left, right, on = ['days since jun 2012'], how = 'outer'), dfs3)

dfs3merged.insert(1, "date", value=0)

for i, row in dfs3merged.iterrows():
    dfs3merged.at[i, 'date'] = jan2012 + datetime.timedelta(days = i)


# Forming exp2 plot
f2, ax2 = plt.subplots(3)
f2.suptitle('Experiment 2 Data')

ax2[0].plot(dfs2merged['date'], dfs2merged['steam rate (t/d)'], 'go', label = 'Steam Rate (t/d)')

ax2[1].plot(dfs2merged['date'], dfs2merged['water rate (m^3/day)'], 'bo', label = 'Water Rate (m^3/day)')
ax2[1].plot(dfs2merged['date'], dfs2merged['oil rate (m^3/day)'], 'yo', label = 'oil rate (m^3/day)')

ax2[2].plot(dfs2merged['date'], dfs2merged['oil rate (m^3/day)'], 'yo', label = 'oil rate (m^3/day)')


plt.show()




pass




    

