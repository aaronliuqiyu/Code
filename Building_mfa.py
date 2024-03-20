#%%
# Import required packages #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dynamic_stock_model import DynamicStockModel as DSM

import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd

pd.options.display.float_format = '{:.2f}'.format

from helper import stock_driven
from helper import outflow_cohort
from helper import MI_cohort

# Set path #
path= "C:\\Users\\qiyu\\OneDrive - Chalmers\\Paper 3\\Data"
os.chdir(path)

pd.set_option('mode.chained_assignment', None)

# Load data #
# Read in file with pickle #
pickle_file_path = "C:\\Users\\qiyu\\OneDrive - Chalmers\\Paper 3\\Data\\Building_input.pkl"
buildings = pd.read_pickle(pickle_file_path)
buildings

# Temporary MI #
# Read in input data from excel file #
file_loc = 'MI_temp.xlsx'

# Read in material intensity values #
MF_structure_MI = pd.read_excel(file_loc, sheet_name='MF structure', index_col=0) 
MF_skin_MI = pd.read_excel(file_loc, sheet_name='MF skin', index_col=0) 
MF_space_MI = pd.read_excel(file_loc, sheet_name='MF space', index_col=0) 

# Read in lifetime parameter from excel #
lifetime = pd.read_excel(file_loc, sheet_name='Lifetime', usecols='A:C', index_col=0)
skin_lifetime = pd.read_excel(file_loc, sheet_name='Skin Lifetime', usecols='A:C', index_col=0)
space_lifetime = pd.read_excel(file_loc, sheet_name='Space Lifetime', usecols='A:C', index_col=0)

new_construction = pd.read_excel(file_loc, sheet_name='New construction', usecols='A:B', index_col=0)

detached = buildings.loc[buildings['Building type number'] == 130]
rowhouse = buildings.loc[buildings['Building type number'] == 131]
terraced = buildings.loc[buildings['Building type number'] == 132]
apartment = buildings.loc[buildings['Building type number'] == 133]
smallhouse = buildings.loc[buildings['Building type number'] == 135]

# Calculate the stock #
detached_fs = detached.groupby(["Construction year"])["Usable floor space"].sum()
detached_fs = detached_fs.cumsum()
detached_fs = pd.DataFrame(detached_fs)

rowhouse_fs = rowhouse.groupby(["Construction year"])["Usable floor space"].sum()
rowhouse_fs = rowhouse_fs.cumsum()
rowhouse_fs = pd.DataFrame(rowhouse_fs)

terraced_fs = terraced.groupby(["Construction year"])["Usable floor space"].sum()
terraced_fs = terraced_fs.cumsum()
terraced_fs = pd.DataFrame(terraced_fs)

apartment_fs = apartment.groupby(["Construction year"])["Usable floor space"].sum()
apartment_fs = apartment_fs.cumsum()
apartment_fs = pd.DataFrame(apartment_fs)

smallhouse_fs = smallhouse.groupby(["Construction year"])["Usable floor space"].sum()
smallhouse_fs = smallhouse_fs.cumsum()
smallhouse_fs = pd.DataFrame(smallhouse_fs)

apartment_fs = pd.concat([apartment_fs,new_construction], axis=0)

# SC is stock by corhort, i.e, a matrix #
apartment_sc, apartment_outflow, apartment_inflow =  stock_driven(apartment_fs, lifetime)
apartment_s = apartment_sc.sum(axis=1)

apartment_structure_stock  = MI_cohort(apartment_sc, MF_structure_MI)
apartment_structure_stock.columns = MF_structure_MI.columns

apartment_skin_stock = MI_cohort(apartment_sc, MF_skin_MI)
apartment_skin_stock.columns = MF_skin_MI.columns

apartment_space_stock = MI_cohort(apartment_sc, MF_space_MI)
apartment_space_stock.columns = MF_space_MI.columns

apartment_skin_ren_floor = outflow_cohort(apartment_sc, skin_lifetime)
apartment_space_ren_floor = outflow_cohort(apartment_sc, space_lifetime)

#%%
# Element wise multiplication#

apartment_structure_inflowa = np.array(apartment_inflow) 
apartment_structure_inflow_m = MF_structure_MI.multiply(apartment_structure_inflowa, axis='columns')

apartment_structure_outflow_m = MI_cohort(apartment_outflow, MF_structure_MI)
apartment_skin_ren = MI_cohort(apartment_skin_ren_floor, MF_skin_MI)
apartment_space_ren = MI_cohort(apartment_space_ren_floor, MF_space_MI)

#%%
# Plotting the stacked bar plot using DataFrame's plotting method
stock_plot = apartment_structure_stock.iloc[-28:] /1000
ax = stock_plot.plot(kind='bar', stacked=True, colormap='viridis')

# Adding labels and title
ax.set_ylabel('Stock (ton)')
ax.set_title('Apartment structure material stock')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Show the plot
plt.show()


#%%
# Plotting the stacked bar plot using DataFrame's plotting method
skin_stock_plot = apartment_skin_stock.iloc[-28:] /1000
ax = skin_stock_plot.plot(kind='bar', stacked=True, colormap='viridis')

# Adding labels and title
ax.set_ylabel('Stock (ton)')
ax.set_title('Apartment skin material stock')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Show the plot
plt.show()
#%%
# Plotting the stacked bar plot using DataFrame's plotting method
space_stock_plot = apartment_space_stock.iloc[-28:] /1000
ax = space_stock_plot.plot(kind='bar', stacked=True, colormap='viridis')

# Adding labels and title
ax.set_ylabel('Stock (ton)')
ax.set_title('Apartment space material stock')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Show the plot
plt.show()

#%%
apartment_fs = pd.DataFrame(apartment_fs)
apartment_fs.plot(kind='line', marker='o', label='My Line Plot')

# Add labels and title
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Line Plot from DataFrame with Index')

# Add a legend
plt.legend()

# Show the plot
plt.show()

#%%
sc_plot = pd.DataFrame(hmm).iloc[-28:] / 1000
sc_plot.plot(kind='line', marker='o', label='My Line Plot')

# Add labels and title
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Line Plot from DataFrame with Index')

# Add a legend
plt.legend()

# Show the plot
plt.show()
#%%
import matplotlib.pyplot as plt

inflow_plot = apartment_structure_inflow_m.iloc[-28:] / 1000
outflow_plot = apartment_structure_outflow_m.iloc[-28:] / 1000

cm = 1/2.54  # centimeters in inches
fig, ((ax1, ax2)) = plt.subplots(1, 2, sharey=True, figsize=(19*cm, 10*cm))

inflow_plot.plot(kind='bar', stacked=True, colormap='viridis',ax=ax1, legend=False)
outflow_plot.plot(kind='bar', stacked=True, colormap='viridis',ax=ax2, legend=True)

ax1.set_title('Apartment structure inflow',fontsize=10)
ax2.set_title('Apartment structure outflow',fontsize=10)

# Adding labels and title
ax1.set_ylabel('Flow (ton)')

ax2.legend(loc='upper left', bbox_to_anchor=(1, 1),fontsize=8)
# Adjust layout
plt.tight_layout()

plt.savefig("Inflow outflow.png",bbox_inches='tight', dpi=800)

# Show the plot
plt.show()

#%%
import matplotlib.pyplot as plt

inflow_plot = apartment_space_ren.iloc[-28:] / 1000
outflow_plot = apartment_skin_ren.iloc[-28:] / 1000

cm = 1/2.54  # centimeters in inches
fig, ((ax1, ax2)) = plt.subplots(1, 2, sharey=True, figsize=(19*cm, 10*cm))

inflow_plot.plot(kind='bar', stacked=True, colormap='viridis',ax=ax1, legend=False)
outflow_plot.plot(kind='bar', stacked=True, colormap='viridis',ax=ax2, legend=True)

ax1.set_title('Apartment space inflow',fontsize=10)
ax2.set_title('Apartment skin inflow',fontsize=10)

# Adding labels and title
ax1.set_ylabel('Flow (ton)')

ax2.legend(loc='upper left', bbox_to_anchor=(1, 1),fontsize=8)
# Adjust layout
plt.tight_layout()

plt.savefig("Renovation.png",bbox_inches='tight', dpi=800)

# Show the plot
plt.show()
#%%
import matplotlib.pyplot as plt

stock_plot = apartment_structure_stock.iloc[-28:]
inflow_plot = apartment_structure_inflow.iloc[-28:]
outflow_plot = apartment_structure_outflow.iloc[-28:]

cm = 1/2.54  # centimeters in inches
fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, sharey=True, figsize=(14*cm, 10*cm))

stock_plot.plot(kind='bar', stacked=True, colormap='viridis', ax=ax1, legend=False)
inflow_plot.plot(kind='bar', stacked=True, colormap='viridis',ax=ax2, legend=False)
outflow_plot.plot(kind='bar', stacked=True, colormap='viridis',ax=ax3, legend=False)

ax1.set_title('Apartment stock',fontsize=10)
ax2.set_title('Apartment inflow',fontsize=10)
ax3.set_title('Apartment outflow',fontsize=10)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()