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

buildings['County'] = buildings['County'].astype(int)
buildings['Municipality'] = buildings['Municipality'].astype(int)

#def add_zero(x):
#    return f'0{x}' if len(str(x)) == 1 else x
#buildings['Municipality'] = buildings['Municipality'].apply(add_zero)
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

non_energy_MI = pd.read_excel(file_loc, sheet_name='MF non energy', index_col=0) 
light_MI = pd.read_excel(file_loc, sheet_name='MF light', index_col=0) 
deep_MI = pd.read_excel(file_loc, sheet_name='MF deep', index_col=0) 

# Percentage of dwellings that have yet to be renovated #
ren_percent = pd.read_excel(file_loc, sheet_name='Percentage', index_col=0)

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

# MFA at the national level #   
apartment_se_int_new = apartment_kommun_int_new.drop(columns='Code')
apartment_se_int_new = apartment_se_int_new.groupby(["Construction year"]).sum()

sc_se, outflow_se, inflow_se = stock_driven(apartment_se_int_new, lifetime)
sc_se.columns = range(1880, 2051)

# MFA for each kommun #
sc_df = []
out_df= []
in_df = []
s_df = []

for i in codes:
    df = apartment_kommun_int_new.loc[apartment_kommun_int_new['Code'] == i]
    df = df.set_index('Construction year')
    df = df.drop(columns=['Code'])

    sc, outflow, inflow = stock_driven(df, lifetime)
    
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
stock_all_kommun = stock_all_kommun.rename(columns={'index': 'Year', 0: 'Usable floor space'})

sc_all_kommun['Kommun'] = repeated_code

inflow_all_kommun['Kommun'] = repeated_code
inflow_all_kommun = inflow_all_kommun.rename(columns={'index': 'Year', 0: 'Inflow'})

outflow_all_kommun['Kommun'] = repeated_code
outflow_all_kommun = outflow_all_kommun.rename(columns={'index': 'Year'})

#%%
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
structure_stock_all = structure_stock_all.rename(columns={'index': 'Year'}) 

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
skin_stock_all = skin_stock_all.rename(columns={'index': 'Year'})

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
space_stock_all = space_stock_all.rename(columns={'index': 'Year'}) 

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

kommuns = apartment_skin_ren_floor['Kommun']
apartment_skin_ren_floor_ee = apartment_skin_ren_floor.drop(columns='Kommun')
repeat_ren_percent = pd.concat([ren_percent] * 290, ignore_index=False, axis=0)
apartment_skin_ren_floor_ee = apartment_skin_ren_floor_ee * repeat_ren_percent
apartment_skin_ren_floor_ee['Kommun'] = kommuns
#%%
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

apartment_space_ren_floor_ee = apartment_space_ren_floor.iloc[:, list(range(101)) + [-1]]

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
    df2 = df2.set_index('Year')
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
apartment_skin_ren_m = apartment_skin_ren_m.rename(columns={'index': 'Year'}) 

skin_ren_m_ee = []

for i in codes:
    df2 = apartment_skin_ren_floor_ee.loc[apartment_skin_ren_floor_ee['Kommun'] == i]
    kommun = df2['Kommun']
    df2 = df2.drop(columns='Kommun')

    apartment_skin_ren_m_ee  = MI_cohort(df2, MF_skin_MI)
    apartment_skin_ren_m_ee['Kommun'] = kommun

    skin_ren_m_ee.append(apartment_skin_ren_m_ee)
    apartment_skin_ren_m_ee = pd.concat(skin_ren_m_ee, ignore_index=False, axis=0)

apartment_skin_ren_m_ee = apartment_skin_ren_m_ee.reset_index()
apartment_skin_ren_m_ee = apartment_skin_ren_m_ee.rename(columns={'index': 'Year'}) 

skin_ren_m_ne = []

for i in codes:
    df2 = apartment_skin_ren_floor_ee.loc[apartment_skin_ren_floor_ee['Kommun'] == i]
    kommun = df2['Kommun']
    df2 = df2.drop(columns='Kommun')
    df2 = pd.DataFrame(df2)

    apartment_skin_ren_m_ne  = MI_cohort(df2, pd.DataFrame(non_energy_MI))
    apartment_skin_ren_m_ne['Kommun'] = kommun

    skin_ren_m_ne.append(apartment_skin_ren_m_ne)
    apartment_skin_ren_m_ne = pd.concat(skin_ren_m_ne, ignore_index=False, axis=0)

apartment_skin_ren_m_ne = apartment_skin_ren_m_ne.reset_index()
apartment_skin_ren_m_ne = apartment_skin_ren_m_ne.rename(columns={'index': 'Year'}) 

skin_ren_m_light = []

for i in codes:
    df2 = apartment_skin_ren_floor_ee.loc[apartment_skin_ren_floor_ee['Kommun'] == i]
    kommun = df2['Kommun']
    df2 = df2.drop(columns='Kommun')
    df2 = pd.DataFrame(df2)

    apartment_skin_ren_m_light  = MI_cohort(df2, pd.DataFrame(light_MI))
    apartment_skin_ren_m_light['Kommun'] = kommun

    skin_ren_m_light.append(apartment_skin_ren_m_light)
    apartment_skin_ren_m_light = pd.concat(skin_ren_m_light, ignore_index=False, axis=0)

apartment_skin_ren_m_light = apartment_skin_ren_m_light.reset_index()
apartment_skin_ren_m_light = apartment_skin_ren_m_light.rename(columns={'index': 'Year'}) 

skin_ren_m_deep = []

for i in codes:
    df2 = apartment_skin_ren_floor_ee.loc[apartment_skin_ren_floor_ee['Kommun'] == i]
    kommun = df2['Kommun']
    df2 = df2.drop(columns='Kommun')
    df2 = pd.DataFrame(df2)

    apartment_skin_ren_m_deep  = MI_cohort(df2, pd.DataFrame(deep_MI))
    apartment_skin_ren_m_deep['Kommun'] = kommun

    skin_ren_m_deep.append(apartment_skin_ren_m_deep)
    apartment_skin_ren_m_deep = pd.concat(skin_ren_m_deep, ignore_index=False, axis=0)

apartment_skin_ren_m_deep = apartment_skin_ren_m_deep.reset_index()
apartment_skin_ren_m_deep = apartment_skin_ren_m_deep.rename(columns={'index': 'Year'}) 
#%%
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
kommun = apartment_skin_ren_floor["Kommun"]
se_skin_ren_floor = apartment_skin_ren_floor.drop(columns='Kommun')
se_skin_ren_floor = se_skin_ren_floor.sum(axis=1)
se_skin_ren_floor = pd.DataFrame(se_skin_ren_floor)
se_skin_ren_floor = se_skin_ren_floor.reset_index()
se_skin_ren_floor = se_skin_ren_floor.rename(columns={'index': 'Year'}) 
se_skin_ren_floor = se_skin_ren_floor.groupby(["Year"]).sum()

se_floor = stock_all_kommun.drop(columns='Kommun')
se_floor = se_floor.groupby(["Year"]).sum()

se_demo = outflow_all_kommun.drop(columns='Kommun')
se_demo = se_demo.groupby(["Year"]).sum()
se_demo = se_demo.sum(axis=1)
se_demo
#%%
# Plotting #
# Sum at national level #
se_skin_m  = apartment_skin_ren_m.groupby(["Year"]).sum()
se_skin_m = se_skin_m.drop(columns='Kommun')

se_space_m  = apartment_space_ren_m.groupby(["Year"]).sum()
se_space_m = se_space_m.drop(columns='Kommun')

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
se_ne_m  = apartment_skin_ren_m_ne.groupby(["Year"]).sum()
se_ne_m = se_ne_m.drop(columns='Kommun')

se_deep_m  = apartment_skin_ren_m_deep.groupby(["Year"]).sum()
se_deep_m = se_deep_m.drop(columns='Kommun')

ne_plot = se_ne_m.iloc[-28:] / 1000
deep_plot = se_deep_m.iloc[-28:] / 1000

cm = 1/2.54  # centimeters in inches
fig, ((ax1, ax2)) = plt.subplots(1, 2, sharey=True, figsize=(19*cm, 10*cm))

ne_plot.plot(kind='bar', stacked=True, colormap='viridis',ax=ax1, legend=False)
deep_plot.plot(kind='bar', stacked=True, colormap='cool',ax=ax2, legend=True)

ax1.set_title('Apartment non energy',fontsize=10)
ax2.set_title('Apartment deep energy',fontsize=10)

# Adding labels and title
ax1.set_ylabel('Flow (ton)')

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(handles1 + handles2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1, 1),fontsize=8)

#ax1.legend(loc='upper left', bbox_to_anchor=(1, 1),fontsize=4)
#ax2.legend(loc='upper left', bbox_to_anchor=(1, 1),fontsize=8)
# Adjust layout
plt.tight_layout()
plt.savefig("Renovation material.png",bbox_inches='tight', dpi=800)
plt.show()

#%%

light_plot = se_light_m.iloc[-48:] / 1000
light_plot.plot(kind='bar', stacked=True, colormap='viridis',legend=False)

plt.legend(loc='upper left', bbox_to_anchor=(1, 1),fontsize=8)
plt.show()
#%%
demo_m  = demolition_m.groupby(["Construction year"]).sum()
demo_m = demo_m.drop(columns='Kommun')

demo_plot = demo_m.iloc[-28:] / 1000
demo_plot.plot(kind='bar', stacked=True, colormap='viridis',legend=False)

plt.legend(loc='upper left', bbox_to_anchor=(1, 1),fontsize=8)
plt.show()

#%%
structure_m  = structure_stock_all.groupby(["Construction year"]).sum()
structure_m = structure_m.drop(columns='Kommun')

structure_plot = structure_m.iloc[-28:] / 1000
structure_plot.plot(kind='area', stacked=True, colormap='viridis',legend=False)

plt.legend(loc='upper left', bbox_to_anchor=(1, 1),fontsize=8)
plt.tight_layout()
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
se_skin_m  = apartment_skin_ren_m.groupby(["Year"]).sum()
se_skin_m = se_skin_m.drop(columns='Kommun')

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
deep_e = se_deep_m * ef
deep_e = deep_e.dropna(axis=1, how='any')
deep_e_plot = deep_e.iloc[-51:] / 1000

deep_e_plot.plot(kind='bar', stacked=True, colormap='viridis',legend=False)

plt.legend(loc='upper left', bbox_to_anchor=(1, 1),fontsize=8)
#plt.set_title('Skin renovation embodied CO2',fontsize=10)
#plt.set_ylabel('CO2 (ton)')
plt.tight_layout()
plt.show()

#%%
se_ne_m  = apartment_skin_ren_m_ne.groupby(["Year"]).sum()
se_ne_m = se_ne_m.drop(columns='Kommun')

se_deep_m  = apartment_skin_ren_m_deep.groupby(["Year"]).sum()
se_deep_m = se_deep_m.drop(columns='Kommun')

ne_e = se_ne_m  * ef
ne_e = ne_e.dropna(axis=1, how='any')
ne_e_plot = ne_e.iloc[-31:] / 1000

deep_e = se_deep_m * ef
deep_e = deep_e.dropna(axis=1, how='any')
deep_e_plot = deep_e.iloc[-31:] / 1000

cm = 1/2.54  # centimeters in inches
fig, ((ax1, ax2)) = plt.subplots(1, 2, sharey=True, figsize=(19*cm, 10*cm))

ne_e_plot.plot(kind='area', stacked=True, colormap='viridis',ax=ax1, legend=False)
deep_e_plot.plot(kind='area', stacked=True, colormap='cool',ax=ax2, legend=True)

ax1.set_title('Apartment non energy emissions',fontsize=10)
ax2.set_title('Apartment deep energy emissions',fontsize=10)

# Adding labels and title
ax1.set_ylabel('Embodied carbon (ton)')

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(handles1 + handles2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1, 1),fontsize=8)

# Adjust layout
plt.tight_layout()
plt.savefig("Renovation emission.png",bbox_inches='tight', dpi=800)
plt.show()
# %%
def label(column):
    if column.name < 1965:
        return '-1965'
    elif 1965 <= column.name <= 1975:
        return '1965-1975'
    elif 1975 <= column.name <= 1985:
        return '1975-1985'
    elif 1985 <= column.name <= 1995:
        return '1985-1995'
    elif 1995 <= column.name <= 2005:
        return '1995-2005'
    elif 2005 <= column.name <= 2015:
        return '1995-2015'
    else:
        return '2015-'

ranges = {
    '-1965': range(1880, 1965),
    '1965-1975': range(1965, 1975),
    '1976-1985': range(1976, 1985),
    '1986-1995': range(1986, 1995),
    '1996-2005': range(1996, 2005),
    '2006-2015': range(2006, 2015),
    '2016-2022': range(2016, 2022),
    '2022-2035': range(2023, 2035),
    '2036-2050': range(2036, 2051),

}

sums = {}
# Iterate over each range
for range_name, columns in ranges.items():
    # Extract columns within the current range
    columns_in_range = sc_se.loc[:, columns[0]:columns[-1]]
    # Sum the values in these columns
    range_sum = columns_in_range.sum(axis=1)
    # Store the sum in the dictionary
    sums[range_name] = range_sum

# Convert the dictionary to DataFrame
sum_df = pd.DataFrame(sums)

sum_df.iloc[-131:].plot(kind='area', stacked=True, colormap='viridis',legend=True)

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Values')
plt.title('Stacked Area Plot')

# Show plot
plt.show()

# %%
structure_m  = structure_stock_all.groupby(["Year"]).sum()
structure_m = structure_m.drop(columns='Kommun')

skin_m  = skin_stock_all.groupby(["Year"]).sum()
skin_m = skin_m.drop(columns='Kommun')

space_m  = space_stock_all.groupby(["Year"]).sum()
space_m = space_m.drop(columns='Kommun')

all_m = pd.concat([structure_m, skin_m, space_m], axis=1).fillna(0)
all_m = structure_m.add(skin_m, fill_value=0).add(space_m, fill_value=0)

all_m.iloc[-131:].plot(kind='area', stacked=True, legend=True)
# Add labels and title
plt.xlabel('Year')
plt.ylabel('Values')
plt.title('Stacked Area Plot')

# Show plot
plt.show()
#%%
cm = 1/2.54  # centimeters in inches
fig, ((ax1, ax2)) = plt.subplots(1, 2, sharey=False, figsize=(19*cm, 10*cm))

floorstock_plot = sum_df.iloc[-131:]
all_m_plot = all_m.iloc[-131:] / 1000000000

floorstock_plot.plot(kind='area', stacked=True, colormap='viridis',ax=ax1, legend=True)
all_m_plot.plot(kind='area', stacked=True, colormap='viridis',ax=ax2, legend=True)

ax1.set_title('Building stock',fontsize=10)
ax2.set_title('Material stock',fontsize=10)

# Adding labels and title
ax1.set_xlabel('Year')
ax1.set_ylabel('Floor space (m2)')
ax2.set_ylabel('Mton')
ax1.legend(loc='upper left', bbox_to_anchor=(1, 1),fontsize=5)
ax2.legend(loc='upper left', bbox_to_anchor=(1, 1),fontsize=5)
# Adjust layout
plt.tight_layout()
plt.savefig("stock area plot.svg", dpi=1200,bbox_inches='tight')
plt.show()
#%%
shapefile_path = "./Kommun_Sweref99TM.shp"
# Read the shapefile
kommun_shape = gpd.read_file(shapefile_path)
kommun_shape['KnKod'] = kommun_shape['KnKod'].astype(int)
kommun_shape = kommun_shape.rename(columns={'KnKod': 'Code'})

def add_zero(x):
    return f'0{x}' if len(str(x)) == 1 else x
buildings['Municipality'] = buildings['Municipality'].apply(add_zero)
buildings['Code'] = buildings['County'].astype(str) + buildings['Municipality'].astype(str)
buildings['Code'] = buildings['Code'].astype(int)

apartment = buildings.loc[buildings['Building type number'] == 133]
apartment = apartment.loc[apartment['Construction year'] >= 1880]

floor = apartment.groupby('Code')['Usable floor space'].sum()
floor = floor.reset_index()
floor['Code'] = floor['Code'].astype(int)
floor = kommun_shape.merge(floor, on='Code')

stockholm_floor = floor.loc[(floor['Code'] == 180) | (floor['Code'] == 126) | (floor['Code'] == 123)| (floor['Code'] == 184)| (floor['Code'] == 163)| (floor['Code'] == 127)| (floor['Code'] == 136)| (floor['Code'] == 138)| (floor['Code'] == 183)| (floor['Code'] == 182)| (floor['Code'] == 162)]

structure_2022 = structure_stock_all.loc[(structure_stock_all['Year'] == 2022)]
skin_2022 = skin_stock_all.loc[(skin_stock_all['Year'] == 2022)]
space_2022 = space_stock_all.loc[(space_stock_all['Year'] == 2022)]

file_loc = 'Code.xlsx'
# Read in material intensity values #
kommun_codes = pd.read_excel(file_loc, index_col=0) 

structure_2022_sum = structure_2022.iloc[:, 1:16].sum(axis=1, skipna=True)
structure_2022_sum = structure_2022_sum.reset_index().drop('index', axis=1)
structure_2022_sum['Kommun'] = kommun_codes
skin_2022_sum = skin_2022.iloc[:, 1:16].sum(axis=1, skipna=True)
skin_2022_sum = skin_2022_sum.reset_index().drop('index', axis=1)
skin_2022_sum['Kommun'] = kommun_codes
space_2022_sum = space_2022.iloc[:, 1:16].sum(axis=1, skipna=True)
space_2022_sum = space_2022_sum.reset_index().drop('index', axis=1)
space_2022_sum['Kommun'] = kommun_codes

structure_sum = structure_2022_sum.rename(columns={0: 'Material stock', 'Kommun':'Code'})
#structure_sum['Code'] = sorted_df['Code'].astype(int)
structure_sum = structure_sum.sort_values(by='Code')
structure_sum = kommun_shape.merge(structure_sum, on='Code')
stockholm_structure_sum = structure_sum.loc[(structure_sum['Code'] == 180) | (structure_sum['Code'] == 126) | (structure_sum['Code'] == 123)| (structure_sum['Code'] == 184)| (structure_sum['Code'] == 163)| (structure_sum['Code'] == 127)| (structure_sum['Code'] == 136)| (structure_sum['Code'] == 138)| (structure_sum['Code'] == 183)| (structure_sum['Code'] == 182)| (structure_sum['Code'] == 162)]

all_sum = structure_2022_sum + skin_2022_sum + space_2022_sum
all_sum['Kommun'] = all_sum['Kommun']/3
all_sum = all_sum.rename(columns={0: 'Material stock', 'Kommun':'Code'})
#all_sum['Code'] = sorted_df['Code'].astype(int)
all_sum = kommun_shape.merge(all_sum, on='Code')
all_sum['Material stock'] = all_sum['Material stock']/1000000000 #kg to mt
all_sum['Area'] = all_sum.area/100000
all_sum['Material stock density'] = all_sum['Material stock'] * 1000 / all_sum['Area']

stockholm_2022_sum = all_sum.loc[(all_sum['Code'] == 180) | (all_sum['Code'] == 126) | (all_sum['Code'] == 123)| (all_sum['Code'] == 184)| (all_sum['Code'] == 163)| (all_sum['Code'] == 127)| (all_sum['Code'] == 136)| (all_sum['Code'] == 138)| (all_sum['Code'] == 183)| (all_sum['Code'] == 182)| (all_sum['Code'] == 162)]

#%%
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

fig, axs = plt.subplots(1, 3, figsize=(15, 10))

# Plot the full datasets
floor.plot("Usable floor space",categorical=False, legend=True, vmin=0,  legend_kwds={"label": "Usable floor space"},ax=axs[0])
all_sum.plot("Material stock",categorical=False, legend=True,  legend_kwds={"label": "Mton"},ax=axs[1])
all_sum.plot("Material stock density",categorical=False, legend=True, legend_kwds={"label": "Kton per km2"}, ax=axs[2])

axs_zoom1 = plt.axes([0.22, 0.41, 0.1, 0.1])  # Adjust position and size of the zoomed-in plot
axs_zoom2 = plt.axes([0.495, 0.41, 0.1, 0.1])  # Adjust position and size of the zoomed-in plot
axs_zoom3 = plt.axes([0.7675, 0.41, 0.1, 0.1])  # Adjust position and size of the zoomed-in plot

# Plot the zoomed-in areas
stockholm_floor.plot("Usable floor space", categorical=False, legend=False, vmin=0,ax=axs_zoom1)
stockholm_2022_sum.plot("Material stock",categorical=False, legend=False,ax=axs_zoom2)
stockholm_2022_sum.plot("Material stock density",categorical=False, legend=False,ax=axs_zoom3)

# Draw a line from the zoomed-in area to the smaller plot
#line = axs[0].plot([zoom_centroid.x, 0.9], [zoom_centroid.y, 0.9], color='green')

axs[0].set_title('Total usable floor space',fontsize=10)
axs[1].set_title('Material stock',fontsize=10)
axs[2].set_title('Material stock density',fontsize=10)
axs_zoom1.set_title('Stockholm',fontsize=10)
axs_zoom2.set_title('Stockholm',fontsize=10)
axs_zoom3.set_title('Stockholm',fontsize=10)

axs[0].axes.xaxis.set_ticklabels([])
axs[0].axes.yaxis.set_ticklabels([])
axs[0].set_axis_off()

axs[1].axes.xaxis.set_ticklabels([])
axs[1].axes.yaxis.set_ticklabels([])
axs[1].set_axis_off()

axs[2].axes.xaxis.set_ticklabels([])
axs[2].axes.yaxis.set_ticklabels([])
axs[2].set_axis_off()

axs_zoom1.axes.xaxis.set_ticklabels([])
axs_zoom1.axes.yaxis.set_ticklabels([])
axs_zoom1.set_axis_off()

axs_zoom2.axes.xaxis.set_ticklabels([])
axs_zoom2.axes.yaxis.set_ticklabels([])
axs_zoom2.set_axis_off()

axs_zoom3.axes.xaxis.set_ticklabels([])
axs_zoom3.axes.yaxis.set_ticklabels([])
axs_zoom3.set_axis_off()
# Adjust layout
#plt.tight_layout()

def annotate_upper_left(ax, label):
    ax.annotate(label, xy=(0, 1.1), xycoords='axes fraction', fontsize=10,
                horizontalalignment='left', verticalalignment='top')
annotate_upper_left(axs[0], '(a)')
annotate_upper_left(axs[1], '(b)')
annotate_upper_left(axs[2], '(c)')


plt.savefig("stock map plot.svg", dpi=1200,bbox_inches='tight')
plt.show()
# %%
