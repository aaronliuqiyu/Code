# This file contains helper functions that is used in the main script #
# The main purpose of having functions is to make the main script more readable and easier to modify #

# Packages need to be imported here and not the main script #
import pandas as pd
import numpy as np

# Import ODYM dynamic stock model #
from dynamic_stock_model import DynamicStockModel as DSM

def stock_driven_init(stock,lifetime, switchtime):  
    shape_list = lifetime.iloc[:, 0]
    scale_list = lifetime.iloc[:, 1]

    time = list(stock.index)

    DSMforward = DSM(t=time, s=np.array(stock).reshape(len(stock),),
                     lt={'Type': 'Weibull', 'Shape': np.array(shape_list), 'Scale': np.array(scale_list)})
    
    init_stock = stock.iloc[:switchtime, :]
    init_stock = np.array(init_stock)
    init_stock = init_stock.reshape((len(init_stock),))

    SwitchTime = len(init_stock) + 1

    out_sc, out_oc, out_i = DSMforward.compute_stock_driven_model_initialstock(init_stock, SwitchTime, NegativeInflowCorrect=True)

    # sum up the total outflow and stock and add years as index #
    out_oc[out_oc < 0] = 0
    #out_oc = out_oc.sum(axis=1)
    out_oc = pd.DataFrame(out_oc, index=np.unique(list(stock.index)))
    #out_sc = out_sc.sum(axis=1)
    out_sc = pd.DataFrame(out_sc, index=np.unique(list(stock.index)))
    out_i = pd.DataFrame(out_i, index=np.unique(list(stock.index)))

    return out_sc, out_oc, out_i

# define a function for calculating the stock driven inflow and outflow
def stock_driven(stock,lifetime):  # stock is the different type of roads in different regions
    shape_list = lifetime.iloc[:, 0]
    scale_list = lifetime.iloc[:, 1]

    DSMforward = DSM(t=list(stock.index), s=np.array(stock),
                     lt={'Type': 'Weibull', 'Shape': np.array(shape_list), 'Scale': np.array(scale_list)})

    out_sc, out_oc, out_i = DSMforward.compute_stock_driven_model(NegativeInflowCorrect=True)

    # sum up the total outflow and stock and add years as index #
    out_oc[out_oc < 0] = 0
    #out_oc = out_oc.sum(axis=1)
    out_oc = pd.DataFrame(out_oc, index=np.unique(list(stock.index)))
    #out_sc = out_sc.sum(axis=1)
    out_sc = pd.DataFrame(out_sc, index=np.unique(list(stock.index)))
    out_i = pd.DataFrame(out_i, index=np.unique(list(stock.index)))

    return out_sc, out_oc, out_i

# define a function for calculating the stock driven inflow and outflow
def stock_driven_norm(stock,lifetime):  # stock is the different type of roads in different regions
    StdDev_list = lifetime.iloc[:, 0]
    mean_list = lifetime.iloc[:, 1]


    DSMforward = DSM(t=list(stock.index), s=np.array(stock),
                     lt={'Type': 'Normal', 'StdDev': np.array(StdDev_list), 'Mean': np.array(mean_list)})

    out_sc, out_oc, out_i = DSMforward.compute_stock_driven_model(NegativeInflowCorrect=True)

    # sum up the total outflow and stock and add years as index #
    out_oc[out_oc < 0] = 0
    out_oc = out_oc.sum(axis=1)
    out_oc = pd.DataFrame(out_oc, index=np.unique(list(stock.index)))
    #out_sc = out_sc.sum(axis=1)
    out_sc = pd.DataFrame(out_sc, index=np.unique(list(stock.index)))
    out_i = pd.DataFrame(out_i, index=np.unique(list(stock.index)))

    return out_oc

def outflow_cohort(stock,lifetime):
    # Initialize an empty list to store DataFrames
    dfs = []

# Apply the function using a for loop
    for i in range(0,201):
        df1 = stock.iloc[:, i]
        result_oc = stock_driven_norm(df1, lifetime)
        dfs.append(result_oc.sum(axis=1))

    # Concatenate all the resulting DataFrames
    final_result = pd.concat(dfs, ignore_index=True, axis=1)
    #final_result = final_result.sum(axis=1)
    final_result = pd.DataFrame(final_result)

    return final_result

def MI_cohort(stock, MI):
    # Initialize an empty list to store DataFrames
    dfs = []

# Apply the function using a for loop
    for i in range(0,len(MI.columns)):
        df1 = MI.iloc[:, i]
        MI_cohort = np.tile(df1, (len(MI), 1))
        result = np.multiply(stock, MI_cohort)
        dfs.append(result.sum(axis=1))

    # Concatenate all the resulting DataFrames
    final_result = pd.concat(dfs, ignore_index=True, axis=1)
    final_result = pd.DataFrame(final_result)
    final_result.columns = MI.columns

    return final_result