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

buildings['County'] = buildings['County'].astype(int)
buildings['Municipality'] = buildings['Municipality'].astype(int)
buildings['Code'] = buildings['County'].astype(str) + buildings['Municipality'].astype(str)
buildings['Code'] = buildings['Code'].astype(int)

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

# Select only apartments #
apartment = buildings.loc[buildings['Building type number'] == 133]

apartment_fs = apartment.groupby(["Construction year", "Code"])["Usable floor space"].sum()
apartment_fs = apartment_fs.cumsum()
apartment_fs = pd.DataFrame(apartment_fs)