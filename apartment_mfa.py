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
file_loc = 'MI_temp1.xlsx'

# Read in material intensity values #
MF_structure_MI = pd.read_excel(file_loc, sheet_name='MF structure', index_col=0) 
MF_skin_MI = pd.read_excel(file_loc, sheet_name='MF skin', index_col=0) 
MF_space_MI = pd.read_excel(file_loc, sheet_name='MF space', index_col=0) 

file_loc1 = 'New Construction.xlsx'

new_construction = pd.read_excel(file_loc1, sheet_name='Sheet1', index_col=0)
kommun_list =  pd.read_excel(file_loc1, sheet_name='Sheet2')
repeated_df = pd.DataFrame(np.repeat(kommun_list.values, 28, axis=0), columns=kommun_list.columns)

file_loc2 = 'Emission factor.xlsx'
ef = pd.read_excel(file_loc2, index_col=0)

# Reshape the new construction data #
new_construction = new_construction.set_index('Code')
repeated_index = np.tile(new_construction.index, len(new_construction.columns))
appended_series = pd.concat([new_construction[col] for col in new_construction.columns])
appended_series.index = repeated_index
new_construction = appended_series.reset_index()
new_construction_kommun = pd.concat([new_construction, repeated_df], axis=1)
new_construction_kommun = new_construction_kommun.rename(columns={'index': 'Construction year', 0: 'Usable floor space'})
new_construction_kommun = new_construction_kommun[['Construction year', 'Code', 'Usable floor space']]

# Read in lifetime parameter from excel #
lifetime = pd.read_excel(file_loc, sheet_name='Lifetime', usecols='A:C', index_col=0)
skin_lifetime = pd.read_excel(file_loc, sheet_name='Skin Lifetime', usecols='A:C', index_col=0)
space_lifetime = pd.read_excel(file_loc, sheet_name='Space Lifetime', usecols='A:C', index_col=0)

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

# Give all kommun all years and interpolate missing years #
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

# Concat the reshaped new construction data #
new = []
for i in codes:
    df = apartment_kommun_int.loc[apartment_kommun_int['Code'] == i]
    new_c = new_construction_kommun.loc[new_construction_kommun['Code'] == i]
    new_c['Usable floor space'] = new_c['Usable floor space'].cumsum()
    new_c['Usable floor space'] = new_c['Usable floor space'] + df.iloc[-1,-1]

    concat_df = pd.concat([df, new_c], axis=0) 

    new.append(concat_df)
    apartment_kommun_int_new = pd.concat(new, ignore_index=True, axis=0)

# MFA for each kommun #
sc_df = []
out_df= []
in_df = []
s_df = []

for i in codes:
    df = apartment_kommun_int_new.loc[apartment_kommun_int_new['Code'] == i]
    df = df.set_index('Construction year')
    df = df.drop(columns=['Code'])

    sc, outflow, inflow = stock_driven(df, skin_lifetime)
    
    sc_df.append(sc)
    out_df.append(outflow)
    in_df.append(inflow)
    s_df.append(sc.sum(axis=1))

    sc_all_kommun = pd.concat(sc_df, ignore_index=False, axis=0)
    stock_all_kommun = pd.concat(s_df, ignore_index=False, axis=0)
    outflow_all_kommun = pd.concat(out_df, ignore_index=False, axis=0)
    inflow_all_kommun = pd.concat(in_df, ignore_index=False, axis=0)

sc_all_kommun = sc_all_kommun.reset_index()
stock_all_kommun = stock_all_kommun.reset_index()
inflow_all_kommun = inflow_all_kommun.reset_index()
outflow_all_kommun = outflow_all_kommun.reset_index()

# Add kommun code #
def repeat_values(df, n):
    # Repeat each row n times
    repeated_df = df.loc[df.index.repeat(n)].reset_index(drop=True)
    return repeated_df

n = 171 # 143 years #
test1 = pd.DataFrame(codes)
repeated_code = repeat_values(test1, n)
repeated_code

stock_all_kommun['Kommun'] = repeated_code
stock_all_kommun = stock_all_kommun.rename(columns={'index': 'Construction year', 0: 'Usable floor space'})

sc_all_kommun['Kommun'] = repeated_code
sc_all_kommun

inflow_all_kommun['Kommun'] = repeated_code
inflow_all_kommun = inflow_all_kommun.rename(columns={'index': 'Construction year', 0: 'Inflow'})

outflow_all_kommun['Kommun'] = repeated_code
outflow_all_kommun = outflow_all_kommun.rename(columns={'index': 'Construction year'})

# Mutiply MI to each kommun #
structure_stock = []
skin_stock = []
space_stock  = []

# Structure stock #
for i in codes:
    df2 = sc_all_kommun.loc[sc_all_kommun['Kommun'] == i]
    df2 = df2.set_index('index')
    kommun = df2['Kommun']
    df2 = df2.drop(columns='Kommun')

    s_stock = MI_cohort(df2, MF_structure_MI)
    s_stock.columns = MF_structure_MI.columns
    s_stock['Kommun'] = kommun

    structure_stock.append(s_stock)
    structure_stock_all = pd.concat(structure_stock, ignore_index=False, axis=0)

structure_stock_all = structure_stock_all.reset_index()
structure_stock_all = structure_stock_all.rename(columns={'index': 'Construction year'}) 

# Skin stock #
for i in codes:
    df2 = sc_all_kommun.loc[sc_all_kommun['Kommun'] == i]
    df2 = df2.set_index('index')
    kommun = df2['Kommun']
    df2 = df2.drop(columns='Kommun')

    sk_stock = MI_cohort(df2, MF_skin_MI)
    sk_stock.columns = MF_skin_MI.columns
    sk_stock['Kommun'] = kommun

    skin_stock.append(sk_stock)
    skin_stock_all = pd.concat(skin_stock, ignore_index=False, axis=0)

skin_stock_all = skin_stock_all.reset_index()
skin_stock_all = skin_stock_all.rename(columns={'index': 'Construction year'})

# Space stock #
for i in codes:
    df2 = sc_all_kommun.loc[sc_all_kommun['Kommun'] == i]
    df2 = df2.set_index('index')
    kommun = df2['Kommun']
    df2 = df2.drop(columns='Kommun')

    sp_stock = MI_cohort(df2, MF_space_MI)
    sp_stock.columns = MF_space_MI.columns
    sp_stock['Kommun'] = kommun

    space_stock.append(sp_stock)
    space_stock_all = pd.concat(space_stock, ignore_index=False, axis=0)

space_stock_all = space_stock_all.reset_index()
space_stock_all = space_stock_all.rename(columns={'index': 'Construction year'}) 

#%%
# Renovation in terms of floor area #
skin_ren = []

for i in codes:
    df2 = sc_all_kommun.loc[sc_all_kommun['Kommun'] == i].copy()
    df2 = df2.set_index('index')
    kommun = df2['Kommun']
    df2 = df2.drop(columns='Kommun')

    apartment_skin_ren  = outflow_cohort(df2, skin_lifetime)
    apartment_skin_ren = apartment_skin_ren.copy()
    apartment_skin_ren['Kommun'] = kommun

    skin_ren.append(apartment_skin_ren)
    apartment_skin_ren_floor = pd.concat(skin_ren, ignore_index=False, axis=0)

space_ren = []

for i in codes:
    df2 = sc_all_kommun.loc[sc_all_kommun['Kommun'] == i].copy()
    df2 = df2.set_index('index')
    kommun = df2['Kommun']
    df2 = df2.drop(columns='Kommun')

    apartment_space_ren  = outflow_cohort(df2, space_lifetime)
    apartment_space_ren = apartment_space_ren.copy()
    apartment_space_ren['Kommun'] = kommun

    space_ren.append(apartment_space_ren)
    apartment_space_ren_floor = pd.concat(space_ren, ignore_index=False, axis=0)

# Multiply MI to floor space #
inflow_sum = inflow_all_kommun.groupby(["Construction year"])["Inflow"].sum()
inflow_sum = inflow_sum.reset_index()
inflow_sum = inflow_sum.set_index('Construction year')

apartment_structure_inflowa = np.array(inflow_sum) 
apartment_structure_inflow_m = MF_structure_MI.multiply(apartment_structure_inflowa, axis='columns')

#%%
demolition = []

for i in codes:
    df2 = outflow_all_kommun.loc[outflow_all_kommun['Kommun'] == i]
    df2 = df2.set_index('Construction year')
    kommun = df2['Kommun']
    df2 = df2.drop(columns='Kommun')

    apartment_structure_structure_m  = MI_cohort(df2, MF_structure_MI)
    apartment_structure_structure_m = apartment_structure_structure_m.copy()
    apartment_structure_structure_m['Kommun'] = kommun


    demolition.append(apartment_structure_structure_m)
    demolition_m = pd.concat(demolition, ignore_index=False, axis=0)

demolition_m = demolition_m.reset_index()

skin_ren_m = []

for i in codes:
    df2 = apartment_skin_ren_floor.loc[apartment_skin_ren_floor['Kommun'] == i]
    kommun = df2['Kommun']
    df2 = df2.drop(columns='Kommun')

    apartment_skin_ren_m  = MI_cohort(df2, MF_skin_MI)
    apartment_skin_ren_m['Kommun'] = kommun

    skin_ren_m.append(apartment_skin_ren_m)
    apartment_skin_ren_m = pd.concat(skin_ren_m, ignore_index=False, axis=0)

apartment_skin_ren_m = apartment_skin_ren_m.reset_index()
apartment_skin_ren_m = apartment_skin_ren_m.rename(columns={'index': 'Construction year'}) 

space_ren_m = []

for i in codes:
    df2 = apartment_space_ren_floor.loc[apartment_space_ren_floor['Kommun'] == i]
    kommun = df2['Kommun']
    df2 = df2.drop(columns='Kommun')

    apartment_space_ren_m  = MI_cohort(df2, MF_space_MI)
    apartment_space_ren_m['Kommun'] = kommun

    space_ren_m.append(apartment_space_ren_m)
    apartment_space_ren_m = pd.concat(space_ren_m, ignore_index=False, axis=0)

apartment_space_ren_m = apartment_space_ren_m.reset_index()
apartment_space_ren_m = apartment_space_ren_m.rename(columns={'index': 'Construction year'}) 
#%%
# Plotting #
# Sum at national level #
se_skin_m  = apartment_skin_ren_m.groupby(["Construction year"]).sum()
se_skin_m = se_skin_m.drop(columns='Kommun')

se_space_m  = apartment_space_ren_m.groupby(["Construction year"]).sum()
se_space_m = se_space_m.drop(columns='Kommun')

import matplotlib.pyplot as plt

inflow_plot = se_space_m.iloc[-28:] / 1000
outflow_plot = se_skin_m.iloc[-28:] / 1000

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
plt.show()

#%%
demo_m  = demolition_m.groupby(["Construction year"]).sum()
demo_m = demo_m.drop(columns='Kommun')

demo_plot = demo_m.iloc[-28:] / 1000
demo_plot.plot(kind='bar', stacked=True, colormap='viridis',legend=False)

plt.legend(loc='upper left', bbox_to_anchor=(1, 1),fontsize=8)
plt.show()
#%%
structure_kommun = structure_stock_all.loc[structure_stock_all['Construction year'] == 2022]
structure_kommun_1 = structure_kommun.loc[(structure_kommun['Kommun'] == 180) | (structure_kommun['Kommun'] == 1480) | (structure_kommun['Kommun'] == 1280) | (structure_kommun['Kommun'] == 380) | (structure_kommun['Kommun'] == 580)]
structure_kommun_1 = structure_kommun_1.set_index('Kommun')
structure_kommun_1 = structure_kommun_1.drop(columns='Construction year')

structure_plot = structure_kommun_1 / 1000
structure_plot.plot(kind='bar', stacked=True, colormap='viridis',legend=False)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1),fontsize=8)

plt.show()
#%%
skin_e = se_skin_m * ef

skin_e_plot = skin_e.iloc[-51:] / 1000

skin_e_plot.plot(kind='bar', stacked=True, colormap='viridis',legend=False)

plt.legend(loc='upper left', bbox_to_anchor=(1, 1),fontsize=8)
#plt.set_title('Skin renovation embodied CO2',fontsize=10)
#plt.set_ylabel('CO2 (ton)')
plt.tight_layout()

plt.savefig("Renovation emission.png",bbox_inches='tight', dpi=800)
plt.show()
#%%
# define a function for calculating the stock driven inflow and outflow
def stock_driven_init(stock,lifetime, init, st):  # stock is the different type of roads in different regions
    shape_list = lifetime.iloc[:, 0]
    scale_list = lifetime.iloc[:, 1]
    
    initial = np.array(init)

    DSMforward = DSM(t=list(stock.index), s=np.array(stock),
                     lt={'Type': 'Weibull', 'Shape': np.array(shape_list), 'Scale': np.array(scale_list)})

    out_sc, out_oc, out_i = DSMforward.compute_stock_driven_model_initialstock(initial, st)

    # sum up the total outflow and stock and add years as index #
    out_oc[out_oc < 0] = 0
    #out_oc = out_oc.sum(axis=1)
    out_oc = pd.DataFrame(out_oc, index=np.unique(list(stock.index)))
    #out_sc = out_sc.sum(axis=1)
    out_sc = pd.DataFrame(out_sc, index=np.unique(list(stock.index)))
    out_i = pd.DataFrame(out_i, index=np.unique(list(stock.index)))

    return out_sc, out_oc, out_i
#%%
def compute_evolution(stock,lifetime, init, st):  # stock is the different type of roads in different regions
    shape_list = lifetime.iloc[:, 0]
    scale_list = lifetime.iloc[:, 1]

    initial = np.array(init)

    DSMforward = DSM(t=list(stock.index), s=np.array(stock),
                     lt={'Type': 'Weibull', 'Shape': np.array(shape_list), 'Scale': np.array(scale_list)})

    out_sc= DSMforward.compute_evolution_initialstock(initial,st)

    out_sc = pd.DataFrame(out_sc, index=np.unique(list(stock.index)))


    return out_sc
#%%
df = apartment_kommun_int_new.loc[apartment_kommun_int_new['Code'] == 126]
df = df.drop(columns=['Code', 'Construction year'])
my_array = np.arange(143)
out_sc, out_oc, out_i = stock_driven_init(df,lifetime, my_array, 144)
# %%
