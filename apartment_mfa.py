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
apartment = apartment.loc[apartment['Construction year'] >= 1880]

# Grouby and sum #
apartment_fs = apartment.groupby(["Code", "Construction year"])["Usable floor space"].sum()
apartment_fs = pd.DataFrame(apartment_fs)
apartment_kommun = apartment_fs.reset_index()

codes = apartment['Code'].unique().astype(int)
min_code = codes.min()
max_code = codes.max()

# Cumsum for each kommun to turn inflow into stock #
final = []
for i in codes:
    df = apartment_kommun.loc[apartment_kommun['Code'] == i]
    df['Usable floor space'] = df['Usable floor space'].cumsum()

    final.append(df)
    apartment_kommun_sum = pd.concat(final, ignore_index=True, axis=0)

apartment_kommun_sum
#%%
years = apartment_kommun_sum['Construction year'].unique().astype(int)
min_year = years.min()
max_year = years.max()
all_years = pd.DataFrame({'Construction year': range(min_year, max_year+1)})

test = []
for i in codes:
    df = apartment_kommun_sum.loc[apartment_kommun_sum['Code'] == i]

    merged_df = pd.merge(all_years, df, on='Construction year', how='left')
    merged_df.iloc[0, -1]  = df.iloc[0, -1]
    merged_df.iloc[-1, -1] = df.iloc[-1, -1]
    # Interpolate missing values
    merged_df['Usable floor space'] = merged_df['Usable floor space'].interpolate()
    merged_df['Code'] = merged_df['Code'].fillna(i.astype(int))

    test.append(merged_df)
    apartment_kommun_int = pd.concat(test, ignore_index=True, axis=0)

apartment_kommun_int
#%%
# MFA for each kommun #
test = []
test1 = []
test2 = []

for i in range(min_year, max_year):
    df = apartment_kommun_int.loc[apartment_kommun_int['Code'] == i]
    sc, outflow, inflow =  stock_driven(df, lifetime)
    
    test.append(sc)
    test1.append(outflow)
    test2.append(inflow)

    final_result = pd.concat(test, ignore_index=True, axis=1)
    final_result = pd.DataFrame(final_result)

final_result
# %%
