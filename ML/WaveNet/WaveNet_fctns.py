import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import copy
from matplotlib import colors
from sklearn import tree
#import skfda
#from sklearn import LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import timeshap
from timeshap.explainer import event_level, feature_level

import seaborn as sns
import scipy
from scipy import stats
from scipy.stats import truncnorm
import scipy.signal
#import cartopy.crs as ccrs
#import cartopy.io.shapereader as shpreader

import torch
import torch.nn as nn
import torch.optim as optim
#import torchvision
#import torch.nn.functional as F
#from torch.utils.data import TensorDataset, DataLoader, random_split
#import torchinfo

import warnings
import os
from copy import deepcopy

import plotting
import dataset_fctns
import modelling_fctns
from ML_fctns import *
import sys
sys.path.append("C:\\Users\\wlwc1989\\Documents\\Phenology_Test_Notebooks\\phenology_dwd\\optimisation_experiments")
from optimise_GDD_fctns import *

from datetime import datetime
from dateutil.relativedelta import relativedelta
#from suntimes import SunTimes  

def split_ds_by_AEZ2(ds):
    AEZ_dict = {1: 'Tropics, lowland; semi-arid',
                2: 'Tropics, lowland; sub-humid',
                3: 'Tropics, lowland; humid',
                4: 'Tropics, highland; semi-arid',
                5: 'Tropics, highland; sub-humid',
                10: 'Sub-tropics, moderately cool; semi-arid',
                11: 'Sub-tropics, moderately cool; sub-humid',
                14: 'Sub-tropics, cool; sub-humid',
                26: 'Land with severe soil/terrain limitations',
                27: 'Land with ample irrigated soils',
                29: 'Desert/Arid climate',
                32: 'Dominantly built-up land',
                33: 'Dominantly water'
                }
    ds_arid_low = ds.loc[ds['AEZ'].isin([1, 26, 29])].groupby(['Stations_id', 'year']).head(2)
    ds_humid_low = ds.loc[ds['AEZ'].isin([2, 3])].groupby(['Stations_id', 'year']).head(2)
    ds_arid_high = ds.loc[ds['AEZ'].isin([4, 10])].groupby(['Stations_id', 'year']).head(2)
    ds_humid_high = ds.loc[ds['AEZ'].isin([5, 11, 14])].groupby(['Stations_id', 'year']).head(2)
    ds_city = ds.loc[ds['AEZ'].isin([32])].groupby(['Stations_id', 'year']).head(2)
    print(f'\nnum arid low: {len(ds_arid_low)}',
        f'\nnum city: {len(ds_city)}',
        f'\nnum arid high: {len(ds_arid_high)}',
        f'\nnum humid low: {len(ds_humid_low)}',
        f'\nnum humid high: {len(ds_humid_high)}')
    ds_dict = {'arid low': ds_arid_low,
               'city': ds_city,
               'arid high': ds_arid_high,
               'humid low': ds_humid_low,
               'humid high': ds_humid_high}
    return ds_dict

def split_ds_by_AEZ5(ds):
    AEZ_dict = {1: 'Tropics, lowland; semi-arid',
                2: 'Tropics, lowland; sub-humid',
                3: 'Tropics, lowland; humid',
                4: 'Tropics, highland; semi-arid',
                5: 'Tropics, highland; sub-humid',
                10: 'Sub-tropics, moderately cool; semi-arid',
                11: 'Sub-tropics, moderately cool; sub-humid',
                14: 'Sub-tropics, cool; sub-humid',
                26: 'Land with severe soil/terrain limitations',
                27: 'Land with ample irrigated soils',
                29: 'Desert/Arid climate',
                32: 'Dominantly built-up land',
                33: 'Dominantly water'
                }
    ds_semiarid_low = ds.loc[ds['AEZ'].isin([1])].groupby(['Stations_id', 'year']).head(2)
    ds_arid_low = ds.loc[ds['AEZ'].isin([26, 29])].groupby(['Stations_id', 'year']).head(2)
    ds_subhumid_low = ds.loc[ds['AEZ'].isin([2])].groupby(['Stations_id', 'year']).head(2)
    ds_humid_low = ds.loc[ds['AEZ'].isin([3])].groupby(['Stations_id', 'year']).head(2)
    ds_arid_high = ds.loc[ds['AEZ'].isin([4])].groupby(['Stations_id', 'year']).head(2)
    ds_humid_high = ds.loc[ds['AEZ'].isin([5])].groupby(['Stations_id', 'year']).head(2)
    ds_cool = ds.loc[ds['AEZ'].isin([10, 11, 14])].groupby(['Stations_id', 'year']).head(2)
    print(#f'\nnum cool: {len(ds_cool)}',
        f'\nnum semiarid low: {len(ds_semiarid_low)}',
        f'\nnum arid high: {len(ds_arid_high)}',
        f'\nnum humid low: {len(ds_humid_low)}',
        f'\nnum subhumid low: {len(ds_subhumid_low)}',
        f'\nnum subhumid high: {len(ds_humid_high)}')
    ds_dict = {#'cool': ds_cool,
               'semiarid low': ds_semiarid_low,
               'arid high': ds_arid_high,
               'humid low': ds_humid_low,
               'subhumid low': ds_subhumid_low,
               'humid high': ds_humid_high}
    return ds_dict

def local_opt(x1s, train, test, error_fctn = run_GDD_and_get_RMSE, lower_bounds = [0.05, 4, 20, 20, 35], upper_bounds = [1, 12, 35, 35, 60], #ds
              great_threshold = 13, response_type = 'Trapezoid', phase_list = ['yellow ripeness'],
              method = 'trust-constr', modified_cost = False, thresholds = [100], growing_period_length = 185,
              maxiter = 50, split = True, CCNN_split = False, test_size = 0.5, random_state=1, itr = 0.5, bias_term = False):
    #if split:
    #    train, test = train_test_split(ds, test_size=test_size, random_state = random_state)
    #    #train_indices = pd.read_csv('C:\\Users\\wlwc1989\\Documents\\Phenology_Test_Notebooks\\phenology_dwd\\results_for_comparing\\model_output\\temp_CCNN_training.csv')
    #elif CCNN_split:
    #    test_indices = pd.read_csv('C:\\Users\\wlwc1989\\Documents\\Phenology_Test_Notebooks\\phenology_dwd\\results_for_comparing\\model_output\\compare_CCNN_results.csv')
    #    train_indices = train_indices.rename(columns = {'year': 'Referenzjahr'})
    #    test_indices = test_indices.rename(columns = {'year': 'Referenzjahr'})
    #    test = pd.merge(ds, test_indices[['Stations_id', 'Referenzjahr']], on = ['Stations_id', 'Referenzjahr'])
    #    complement_indices = ds.index.difference(test.index)
    #
    #    # Filter df2 to keep only the rows with the complement indices
    #    train = ds.loc[complement_indices]
    #    #train = pd.merge(ds, train_indices[['Stations_id', 'Referenzjahr']], on = ['Stations_id', 'Referenzjahr'])
    #    test = pd.merge(ds, test_indices[['Stations_id', 'Referenzjahr']], on = ['Stations_id', 'Referenzjahr'])
    #else:
    #    train = ds
    #    test = ds
    only_phase = phase_list[0]
    if response_type == 'Trapezoid':
        ineq_cons = {'type': 'ineq',
                    'fun' : lambda x: np.array([x[2] - x[1],
                                                x[3] - x[2],
                                                x[4] - x[3]]),
                    'jac' : lambda x: np.array([[0, -1, 1, 0, 0],
                                                [0, 0, -1, 1, 0],
                                                [0, 0, 0, -1, 1]])}
        constraints = scipy.optimize.LinearConstraint([[0, -1, 1, 0, 0], [0, 0, -1, 1, 0], [0, 0, 0, -1, -1]], [0, 0, 0], [np.inf, np.inf, np.inf])
    elif response_type == 'Wang' or response_type == 'Convolved':
        ineq_cons = {'type': 'ineq',
                    'fun' : lambda x: np.array([x[2] - x[1],
                                                x[3] - x[2]]),
                    'jac' : lambda x: np.array([[0, -1, 1, 0],
                                                [0, 0, -1, 1]])}
        ieq_cons = [lambda x: x[2] - x[1],
                    lambda x: x[3] - x[2]]
        if bias_term:
            constraints = scipy.optimize.LinearConstraint(np.array([[0, -1, 1, 0, 0], [0, 0, -1, 1, 0]]),lb= [0, 0], ub=[1000, 1000])
        else:  
            constraints = scipy.optimize.LinearConstraint(np.array([[0, -1, 1, 0], [0, 0, -1, 1]]),lb= [0, 0], ub=[1000, 1000])
    elif response_type == 'Convolved':
        ineq_cons = {'type': 'ineq',
                    'fun' : lambda x: np.array([x[2] - x[1],
                                                x[3] - x[2]]),
                    'jac' : lambda x: np.array([[0, -1, 1, 0],
                                                [0, 0, -1, 1]])}
        ieq_cons = [lambda x: x[2] - x[1],
                    lambda x: x[3] - x[2]]
        constraints = scipy.optimize.LinearConstraint(np.array([[0, -1, 1, 0, 0, 0], [0, 0, -1, 1, 0, 0]]),lb= [0, 0], ub=[1000, 1000])
    elif response_type == 'Spline':
        ineq_cons = {'type': 'ineq',
                    'fun' : lambda x: np.array([x[2],
                                                x[3]]),
                    'jac' : lambda x: np.array([[0 for count in range(20)],
                                                [0 for count in range(20)]])}
        ieq_cons = [lambda x: x[2] - x[1],
                    lambda x: x[3] - x[2]]
        constraints = scipy.optimize.LinearConstraint(np.array([[0 for count in range(20)], [0 for count in range(20)]]),lb= [-1, -1], ub=[1000, 1000])
    elif response_type == 'multi_phase':
        ineq_cons = {'type': 'ineq',
                    'fun' : lambda x: np.array([x[2] - x[1],
                                                x[3] - x[2],
                                                x[6] - x[5],
                                                x[7] - x[6]]
                                                ),
                    'jac' : lambda x: np.array([[0, -1, 1, 0, 0, -1, 1, 0],
                                                [0, 0, -1, 1, 0, 0, -1, 1]])}
        ieq_cons = [lambda x: x[2] - x[1],
                    lambda x: x[3] - x[2]]
        if bias_term:
            constraints = scipy.optimize.LinearConstraint(np.array([[0, -1, 1, 0, 0], [0, 0, -1, 1, 0]]),lb= [0, 0], ub=[1000, 1000])
        else:  
            constraints = scipy.optimize.LinearConstraint(np.array([[0, -1, 1, 0], [0, 0, -1, 1]]),lb= [0, 0], ub=[1000, 1000])
    bounds = scipy.optimize.Bounds(lb=lower_bounds, ub = upper_bounds)
    x0 = np.array([1, 4, 25, 35, 45])
    final_mins = []
    for x0 in x1s:
        if method == 'trust-constr':
            res = scipy.optimize.minimize(lambda x: error_fctn(x, train, 't2m', 
                                                                         response_type = response_type, 
                                                                         phase_list = phase_list, 
                                                                         new_unfinished_penalisation=modified_cost, 
                                                                         growing_period_length = growing_period_length,
                                                                         thresholds = thresholds), 
                                    x0, method = 'trust-constr',#'COBYQA',
                                    #jac='3-point',#
                                    jac = lambda x: run_GDD_and_get_RMSE_derivs(x, train, 't2m', #jac='3-point',#[x0/1000 for x0 in 
                                                                                response_type = response_type, 
                                                                                phase_list = phase_list, 
                                                                                growing_period_length=growing_period_length,
                                                                                thresholds = thresholds),#
                                    constraints=[constraints],
                                    options={'verbose': 1, 'initial_tr_radius': itr, 'xtol':1e-7, 'maxiter':maxiter},# ,'gtol':1e-7, 'finite_diff_rel_step': 0.05
                                    bounds=bounds, tol=1e-9)
        elif method == 'SLSQP':
            res = scipy.optimize.minimize(lambda x: error_fctn(x, train, 't2m', 
                                                                         response_type = response_type, 
                                                                         phase_list = phase_list,
                                                                         new_unfinished_penalisation=modified_cost,
                                                                         growing_period_length=growing_period_length,
                                                                         thresholds = thresholds), 
                                    x0, method = 'SLSQP',
                                    #jac = '3-point',
                                    jac = lambda x: run_GDD_and_get_RMSE_derivs(x, train, 't2m', 
                                                                                response_type = response_type, 
                                                                                phase_list = phase_list, 
                                                                                growing_period_length=growing_period_length,
                                                                                thresholds = thresholds),#jac='3-point',#
                                    constraints=[ineq_cons],
                                    options={'disp': 3, 'maxiter':maxiter, 'ftol': 1e-16},
                                    bounds=bounds)#, tol=1e-15)
        elif method == 'Nelder-Mead':
            res = scipy.optimize.minimize(lambda x: error_fctn(x, train, 't2m', 
                                                                         response_type = response_type, 
                                                                         phase_list = phase_list,
                                                                         new_unfinished_penalisation=modified_cost,
                                                                         growing_period_length=growing_period_length,
                                                                         thresholds = thresholds), 
                                    x0, method = 'Nelder-Mead',
                                    #jac = '3-point',
                                    #jac = lambda x: run_GDD_and_get_RMSE_derivs(x, train, 't2m', 
                                    #                                            response_type = response_type, 
                                    #                                            phase_list = phase_list, 
                                    #                                            growing_period_length=growing_period_length,
                                    #                                            thresholds = thresholds),#jac='3-point',#
                                    options={'disp': True},#, 'maxiter':50
                                    bounds=bounds)#, tol=1e-15)
        elif method == 'Powell':
            res = scipy.optimize.minimize(lambda x: error_fctn(x, train, 't2m', 
                                                                         response_type = response_type, 
                                                                         phase_list = phase_list,
                                                                         new_unfinished_penalisation=modified_cost,
                                                                         growing_period_length=growing_period_length,
                                                                         thresholds = thresholds), 
                                    x0, method = 'Powell',
                                    #jac = '3-point',
                                    #jac = lambda x: run_GDD_and_get_RMSE_derivs(x, train, 't2m', 
                                    #                                            response_type = response_type, 
                                    #                                            phase_list = phase_list, 
                                    #                                            growing_period_length=growing_period_length,
                                    #                                            thresholds = thresholds),#jac='3-point',#
                                    options={'disp': True},#, 'maxiter':50
                                    bounds=bounds)
        print(x0, res.x, x0 - res.x)
        RMSE = error_fctn(res.x, test, 't2m', response_type = response_type, phase_list = phase_list, thresholds = thresholds, growing_period_length = growing_period_length)
        #print('R^2: ', r2_score(c_array[f'observed time to {only_phase}'], c_array[f'modelled time to {only_phase}']))
        print('RMSE at opt on test set: ', RMSE)
        print('R^2:', 1 - (RMSE**2/(test[f'observed time to {only_phase}'].var()*((len(test) - 1)/len(test)))))
        final_mins.append(res.x)
    return final_mins, res, train, test

def train_test_ds_by_year(x_train, y_train, test_years = [2015, 2016, 2017], year_column = -2, train_proportion = 0.8):
    samples_in_test_years = torch.isin(x_train[:, year_column, 0], torch.Tensor(test_years))
    train_ds = TensorDataset(x_train[~samples_in_test_years, :, :], y_train[~samples_in_test_years, :])
    val_ds = TensorDataset(x_train[samples_in_test_years, :, :], y_train[samples_in_test_years, :])
    return train_ds, val_ds

def train_test_dl_by_year(x_train, y_train, bs = 500, n_channels=1, train_proportion = 0.8, test_years = [2015, 2016, 2017], year_column = -2):
    train_ds, val_ds = train_test_ds_by_year(x_train, y_train, train_proportion = train_proportion, test_years = test_years, year_column = year_column)
    train_dl = DataLoader(train_ds, batch_size=bs)
    val_dl = DataLoader(val_ds, batch_size=bs)
    train_dl = WrappedDataLoader(train_dl, lambda x, y: preprocess(x, y, channels =n_channels))
    val_dl = WrappedDataLoader(val_dl, lambda x, y: preprocess(x, y, channels =n_channels))
    return train_dl, val_dl
def mindex_at_0(tens):
    indices = torch.tile(torch.arange(0, tens.shape[1]).float(), (tens.shape[0], 1))  # Shape: (4, 12)
    tens_rounded = torch.round(tens)
    tensmin = torch.min(torch.abs(tens_rounded), dim=1).values
    indices[(torch.abs(tens).T != tensmin).T] = float('nan')
    masked = torch.where(torch.isnan(indices), torch.full_like(indices, float('inf')), indices)
    result = torch.amin(masked, dim=1)##
    # Optionally: Set -inf back to NaN where all were NaN in the column
    result[result == float('inf')] = float('nan')
    return result

def load_model(savename, model):
    model_dir = 'C:\\Users\\wlwc1989\\Documents\\Phenology_Test_Notebooks\\ML_algorithms\\saved_models\\'
    model_path = os.path.join(model_dir, savename + ".pt")
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    #print(checkpoint['model_state_dict'])
    return model

def plot_from_saved(savename, model, val_dl, method = 'regression', 
                    bce=False, CNN=False, title = 'fitted vs. observed', 
                    MMD=False, DTF = False, obs_method=False,
                    modify_u0=False, direct_obs = False, RMSE = False):
    model_dir = 'C:\\Users\\wlwc1989\\Documents\\Phenology_Test_Notebooks\\ML_algorithms\\saved_models\\'
    model_path = os.path.join(model_dir, savename + ".pt")
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    if modify_u0:
        model.u0.data = model.u0.data*3#2.26
    #print(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if method == 'cumulative':
        plot_fitted_observed_cumulative(model, val_dl, bce=bce, CNN=CNN, title=title, MMD=MMD, direct_obs = direct_obs)
    elif method == 'regression':
        plot_fitted_observed_TS(model, val_dl, bce=bce, CNN=CNN, title=title, MMD=MMD, DTF = DTF, obs_method=obs_method, direct_obs = direct_obs, RMSE = RMSE)
    elif method == 'histogram':
        plot_fitted_observed_histogram(model, val_dl, bce=bce, CNN=CNN, title=title, MMD=MMD, DTF = DTF, obs_method=obs_method, direct_obs = direct_obs)
    
def plot_fitted_observed_TS(TS_model, dl, bce=False, CNN=False, title = 'fitted vs. observed', MMD = False, DTF = False, obs_method = False, direct_obs = False, RMSE = False):
    list_logs = []
    list_observed = []
    for xb, yb in dl:
        with torch.no_grad():
            if RMSE:
                list_logs.append(torch.squeeze(TS_model(xb)))# + 4
            elif CNN:
                if MMD:
                    list_logs.append(TS_model(xb)[0])
                else:
                    list_logs.append(TS_model(xb))
            else:
                list_logs.append(TS_model(xb.transpose(1, 2)))
            #print(TS_model(xb.transpose(1, 2)))
        if direct_obs:
            list_observed.append(yb.squeeze())
        else:
            list_observed.append(yb)
    logs = torch.squeeze(torch.cat(list_logs))
    observed = torch.cat(list_observed)
    #print(logs.shape, torch.cat(list_logs, dim=0).shape, observed.shape)
    if RMSE:
        fitted_days = logs
    else:
        if not(DTF):
            if bce:
                fitted = torch.round(logs)
            else:
                fitted = torch.argmax(logs, dim=2)
        if DTF:
            logs = torch.round(logs)
            if not(direct_obs):
                observed_days = mindex_at_0(observed)
            #observed_days = observed[:, 0]
            #observed_days = 90 + 100 - observed[:, 90]
            if obs_method:
                fitted_days = observed_days + logs[range(0, logs.shape[0]), observed_days.int()]
            else:
                fitted_days = mindex_at_0(logs)
                #fitted_days = 90 + 100 - logs[:, 90]
                fitted_days = 90 + logs[:, 90]
            print(torch.sum(fitted_days < 30), ' fits too small')
        else:
            L = fitted.shape[1]
            fitted_days = L - fitted.sum(dim=1)
            if not(direct_obs):
                observed_days = L - observed.sum(dim=1)
    if direct_obs:
        observed_days = observed
    comparison_frame = pd.DataFrame({'fitted': fitted_days.numpy().squeeze(), 'observed': observed_days.numpy().squeeze()})
    #comparison_frame = comparison_frame.loc[comparison_frame['fitted'] <= 90]
    maxval = max(comparison_frame['fitted'].max(), comparison_frame['observed'].max())
    minval = min(comparison_frame['fitted'].min(), comparison_frame['observed'].min())
    fig, ax = plt.subplots()
    sns.regplot(x='fitted', y='observed', data = comparison_frame, ax=ax,
                scatter_kws={'alpha':0.5, 's':4},  x_bins=np.arange(minval - 5, maxval + 5, 3))
    ax.plot([minval, maxval], [minval, maxval], linestyle = '--', color='k')
    ax.set_title(title)
    rsquared = r2_score(comparison_frame['observed'], comparison_frame['fitted'])
    print(f'R^2 value for model: {rsquared}')
    bias = comparison_frame['observed'].mean() - comparison_frame['fitted'].mean()
    variance_modelled = comparison_frame[f'fitted'].var()
    print(f'Bias: {bias**2}\nVariance of modelled values: {variance_modelled}')

def plot_fitted_observed_histogram(TS_model, dl, bce=False, CNN=False, title = 'fitted vs. observed', MMD=False, DTF = False, obs_method = False):
    list_logs = []
    list_observed = []
    for xb, yb in dl:
        with torch.no_grad():
            if CNN:
                if MMD:
                    list_logs.append(TS_model(xb)[0])
                else:
                    list_logs.append(TS_model(xb))
            else:
                list_logs.append(TS_model(xb.transpose(1, 2)))
            #print(TS_model(xb.transpose(1, 2)))
        list_observed.append(yb)
    logs = torch.cat(list_logs).squeeze()
    observed = torch.cat(list_observed)
    #print(logs.shape)
    if bce:
        fitted = torch.round(logs)
    else:
        fitted = torch.argmax(logs, dim=2)
    if DTF:
        logs = torch.round(logs)
        observed_days = mindex_at_0(observed)
        observed_days = observed[:, 0]
        observed_days = 90 + 100 - observed[:, 90]
        if obs_method:
            fitted_days = observed_days + logs[range(0, logs.shape[0]), observed_days.int()]
        else:
            fitted_days = mindex_at_0(logs)
            #fitted_days = 90 + logs[:, 90]
            #fitted_days = 90 + 100 - logs[:, 90]
            fitted_days = 90 + logs[:, 90]
        print(torch.sum(fitted_days < 30), ' fits too small')
    else:
        L = fitted.shape[1]
        fitted_days = L - fitted.sum(dim=1)
        observed_days = L - observed.sum(dim=1)
    comparison_frame = pd.DataFrame({'fitted': fitted_days.numpy().squeeze(), 'observed': observed_days.numpy().squeeze()})
    #comparison_frame = comparison_frame.loc[comparison_frame['fitted'] <= 90]
    maxval = max(comparison_frame['fitted'].max(), comparison_frame['observed'].max())
    minval = min(comparison_frame['fitted'].min(), comparison_frame['observed'].min())
    fig, ax = plt.subplots()
    sns.histplot(x='fitted', data = comparison_frame, ax=ax, label = 'fitted',
                stat = 'density', bins=10)
    sns.histplot(x='observed', data = comparison_frame, ax=ax, label= 'observed',
                stat = 'density')
    ax.set_xlabel('Days to anthesis')
    ax.set_title(title)
    #ax.plot([minval, maxval], [minval, maxval], linestyle = '--', color='k')
    fig.legend(bbox_to_anchor = (1.2, 0.9))
    rsquared = r2_score(comparison_frame['observed'], comparison_frame['fitted'])
    print(f'R^2 value for model: {rsquared}')
    bias = comparison_frame['observed'].mean() - comparison_frame['fitted'].mean()
    variance_modelled = comparison_frame[f'fitted'].var()
    print(f'Bias: {bias**2}\nVariance of modelled values: {variance_modelled}')

def make_error_df(R2s, Biases, Variances, RMSEs, STDs, Corrs, Mins, LQs, Medians, UQs, Maxs, model, region, eval_method):
    error_df = pd.DataFrame({
        'model': [model],
        'region': [region],
        'evaluation method': [eval_method],
        'R2': [np.mean(R2s)],
        'Bias (obs minus modelled)': [np.mean(Biases)],
        'Variance': [np.mean(Variances)],
        'RMSE': [np.mean(RMSEs)],
        'STD': [np.mean(STDs)],
        'Corr': [np.mean(Corrs)],
        'Min': [np.mean(Mins)],
        'LQ': [np.mean(LQs)],
        'Median': [np.mean(Medians)],
        'UQ': [np.mean(UQs)],
        'Max': [np.mean(Maxs)]
    })
    return error_df
def get_ft_params(model, transfer_method):
    if transfer_method == 'KG':
        ft_params = [model.u0, model.u1, model.u2, model.u3, model.p0, model.starting_mask]
    elif transfer_method == 'last_layer':
        ft_params = model.fc.parameters()
    elif transfer_method == 'first_layer':
        ft_params = [param for param in model.conv1.parameters()] + [model.starting_mask]#model.conv1.parameters()#
    elif transfer_method == 'last_conv':
        ft_params = model.conv21.parameters()
    elif transfer_method == 'last_wave':
        ft_params = [layer.parameters() for layer in model.wave3]
        ft_params = [param for layer_params in ft_params for param in layer_params]
    elif transfer_method == 'first_wave':
        ft_params = [layer.parameters() for layer in model.wave1]
        ft_params = [param for layer_params in ft_params for param in layer_params]
    return ft_params

def K_fold_transfer(k_folds, ds, model_class, 
                    savename, epochs, bs, model_args, 
                    transfer_method = 'KG', #could be any of: 'KG', 'last_layer', 'first_layer', 'last_wave', 'first_wave
                    lr = 0.01, savename_prefix = 'KFold', 
                    freeze_params = False, loss = 'NLL', 
                    CNN=False, bce=False, custom_loss = None, 
                    MMD = False, DTF = False, obs_method = False, verbose=False):
    ## LSTM_args: input_dim, hidden_dim, num_layers, output_dim
    # Initialize the k-fold cross validation
    # Loop through each fold
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_values = []
    R2s = []
    Biases = []
    Variances = []
    obsvars = []

    RMSEs = []
    SSEs = []
    STDs = []
    Corrs = []

    Mins = []
    LQs = []
    Medians = []
    UQs = []
    Maxs = []

    lengths = []
    kf = KFold(n_splits=k_folds, shuffle=True, random_state = 1)
    statyear = ds[['Stations_id', 'year']].drop_duplicates()
    for fold, (train_idx, test_idx) in enumerate(kf.split(statyear)):
        train_statyear = statyear.iloc[train_idx]
        test_statyear = statyear.iloc[test_idx]
        train = ds.merge(train_statyear, on=['Stations_id', 'year'], how = 'inner') #.loc[ds[['Stations_id', 'year']].isin(train_statyear)]
        test = ds.merge(test_statyear, on=['Stations_id', 'year'], how = 'inner')
        lengths.append(len(test))
        #print(f"Fold {fold + 1}")
        X_tensor_train, y_tensor_train = ML_tensor_from_ds(train)
        X_tensor_test, y_tensor_test = ML_tensor_from_ds(test)
        #print(len(ds), X_tensor_test.shape, X_tensor_train.shape)
        train_ds, ___ = train_test_ds_from_tensors(X_tensor_train, y_tensor_train, train_proportion=1)
        val_ds, ___ = train_test_ds_from_tensors(X_tensor_test, y_tensor_test, train_proportion=1)
        # Define the data loaders for the current fold
        train_dl = DataLoader(
            dataset=train_ds,
            batch_size=bs,
        )
        val_dl = DataLoader(
            dataset=val_ds,
            batch_size=bs,
        )
        #lenval = 0
        #lentr = 0
        #for xb, yb in train_dl:
        #    lentr += xb.shape[0]
        #for xb, yb in val_dl:
        #    lenval += xb.shape[0]
        #print(lentr, lenval)
        # Initialize the model and optimizer
        model = model_class(*model_args).to(device)
        
        model_dir = 'C:\\Users\\wlwc1989\\Documents\\Phenology_Test_Notebooks\\ML_algorithms\\saved_models\\'
        model_path = os.path.join(model_dir, savename + ".pt")
        checkpoint = torch.load(model_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        #for param in model.fc.parameters():
        #    param.requires_grad = False
        #for param in model.lstm.parameters():
        #    param.requires_grad = False
        ft_params = get_ft_params(model, transfer_method)
        #print(ft_params)
        for param in model.parameters():
            param.requires_grad = False
        for param in ft_params:
            param.requires_grad = True
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        if loss == 'NLL':
            criterion = nn.NLLLoss()
        elif loss == 'BCE':
            criterion = nn.BCELoss()
        elif loss == 'MSE':
            criterion = custom_loss
        # Train the model on the current fold
        if freeze_params:
            for param in [model.u0, model.u1, model.u2, model.u3, model.p0]:
                param.requires_grad = False
        #for param in [model.u0, model.u1, model.u2, model.u3, model.p0]:
        #    print(param.requires_grad, param.shape)
        model_loss = fit_for_kf(epochs, model, criterion, optimizer, 
                                train_dl, val_dl, save_name = savename_prefix + str(fold + 1), 
                                CNN=CNN, bce=bce, plot_opt=False, verbose=verbose)
        loss_values.append(model_loss)

        #Now look at stats for model:
        model_dir = 'C:\\Users\\wlwc1989\\Documents\\Phenology_Test_Notebooks\\ML_algorithms\\saved_models\\'
        model_path = os.path.join(model_dir, savename_prefix + str(fold + 1) + ".pt")
        checkpoint = torch.load(model_path, weights_only=True)
        TS_model = model_class(*model_args).to(device)
        TS_model.load_state_dict(checkpoint['model_state_dict'])

        # Now convert to days and get R^2 to compare to other models
        list_logs = []
        list_observed = []
        for xb, yb in val_dl:
            with torch.no_grad():
                if CNN:
                    if MMD:
                        list_logs.append(TS_model(xb)[0])
                    else:
                        list_logs.append(TS_model(xb))
                else:
                    list_logs.append(TS_model(xb.transpose(1, 2)))
                #print(TS_model(xb.transpose(1, 2)))
            list_observed.append(yb)
        logs = torch.squeeze(torch.cat(list_logs))
        observed = torch.cat(list_observed)
        #print(logs.shape, torch.cat(list_logs, dim=0).shape)
        if not(DTF):
            if bce:
                fitted = torch.round(logs)
            else:
                fitted = torch.argmax(logs, dim=2)
        if DTF:
            logs = torch.round(logs)
            observed_days = mindex_at_0(observed)
            observed_days = observed[:, 0]
            observed_days = 90 + 100 - observed[:, 90]
            if obs_method:
                fitted_days = observed_days + logs[range(0, logs.shape[0]), observed_days.int()]
            else:
                fitted_days = mindex_at_0(logs)
                fitted_days = 90 + 100 - logs[:, 90]
                #fitted_days = logs[:, 90]
            #print(torch.sum(fitted_days < 30), ' fits too small')
        else:
            L = fitted.shape[1]
            fitted_days = L - fitted.sum(dim=1)
            observed_days = L - observed.sum(dim=1)
        
        comparison_frame = pd.DataFrame({'fitted': fitted_days.numpy().squeeze(), 'observed': observed_days.numpy().squeeze()})
        comparison_frame['Error'] = comparison_frame['observed'] - comparison_frame['fitted']

        bias_model = 0#comparison_frame['Error'].mean()
        RMSE = np.sqrt(np.mean((comparison_frame['Error'] - bias_model)**2))
        SSE = np.sum((comparison_frame['Error'] - bias_model)**2)
        STD = comparison_frame['fitted'].std() 
        Corr = comparison_frame['observed'].corr(comparison_frame['fitted'])

        RMSEs.append(RMSE)
        SSEs.append(SSE)
        STDs.append(STD)
        Corrs.append(Corr)
        
        Mins.append(comparison_frame['Error'].min())
        LQs.append(comparison_frame['Error'].quantile(0.25))
        Medians.append(comparison_frame['Error'].median())
        UQs.append(comparison_frame['Error'].quantile(0.75))
        Maxs.append(comparison_frame['Error'].max())

        rsquared = r2_score(comparison_frame['observed'], comparison_frame['fitted'])
        bias = comparison_frame['observed'].mean() - comparison_frame['fitted'].mean()
        variance_modelled = comparison_frame[f'fitted'].var()
        variance_observed = comparison_frame['observed'].var()
        
        R2s.append(rsquared)
        Biases.append(bias)
        Variances.append(variance_modelled)
        obsvars.append(variance_observed)
        if fold == 0:
            comparison_frame_full = comparison_frame
        else:
            comparison_frame_full = pd.concat((comparison_frame, comparison_frame_full))
        #fig, ax = plt.subplots()
        #sns.regplot(x='fitted', y='observed', data = comparison_frame,
        #            scatter_kws={'alpha':0.5, 's':4},  x_bins=10)
        #maxval = max(comparison_frame['fitted'].max(), comparison_frame['observed'].max())
        #minval = min(comparison_frame['fitted'].min(), comparison_frame['observed'].min())
        #ax.plot([minval, maxval], [minval, maxval], linestyle = '--', color='k')

    return comparison_frame_full, loss_values, R2s, Biases, Variances, RMSEs, SSEs, STDs, Corrs, Mins, LQs, Medians, UQs, Maxs, lengths, obsvars

def K_fold_crossval(k_folds, train_ds, model_class, 
                    epochs, bs, model_args, savename_prefix = 'KFold', 
                    freeze_params = False, loss = 'NLL', 
                    CNN=False, bce=False, lr = 0.01, 
                    MMD=False, n_channels = 4, DTF = False, 
                    obs_method = False, custom_loss = None,
                    year_split = False, year_folds = None,
                    GDD_init = None, verbose = False):
    ## LSTM_args: input_dim, hidden_dim, num_layers, output_dim
    # Initialize the k-fold cross validation
    index_cols = ['Stations_id', 'year', 'Management', 'vargroup']
    kf = KFold(n_splits=k_folds, shuffle=True, random_state = 1)
    # Loop through each fold
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_values = []
    R2s = []
    Biases = []
    Variances = []

    RMSEs = []
    STDs = []
    Corrs = []

    Mins = []
    LQs = []
    Medians = []
    UQs = []
    Maxs = []
    if year_split:
        splitter = year_folds
    else:
        statyear = train_ds[index_cols].drop_duplicates()
        splitter = kf.split(statyear)
    for fold, traintest_idx in enumerate(splitter):
        print(f"Fold {fold + 1}")
        if year_split:
            X_tensor, y_tensor = train_ds
            train_dl, val_dl = train_test_dl_by_year(X_tensor, y_tensor, test_years = traintest_idx, year_column = -2, bs = bs, n_channels = 8)
        # Define the data loaders for the current fold
        else:
            train_idx, test_idx = traintest_idx
            train_statyear = statyear.iloc[train_idx]
            test_statyear = statyear.iloc[test_idx]
            train = train_ds.merge(train_statyear, on=index_cols, how = 'inner') #.loc[ds[['Stations_id', 'year']].isin(train_statyear)]
            test = train_ds.merge(test_statyear, on=index_cols, how = 'inner')
            X_tensor_train, y_tensor_train = ML_tensor_from_ds(train)
            X_tensor_test, y_tensor_test = ML_tensor_from_ds(test)
            #print(len(ds), X_tensor_test.shape, X_tensor_train.shape)
            train_ds, ___ = train_test_ds_from_tensors(X_tensor_train, y_tensor_train, train_proportion=1)
            val_ds, ___ = train_test_ds_from_tensors(X_tensor_test, y_tensor_test, train_proportion=1)
            train_dl = DataLoader(
                dataset=train_ds,
                batch_size=bs,
            )
            val_dl = DataLoader(
                dataset=val_ds,
                batch_size=bs,
            )
        print(len(train_dl.dl.dataset), len(val_dl.dl.dataset))
        # Initialize the model and optimizer
        model = model_class(*model_args).to(device)
        if GDD_init is not None:
            if GDD_init == 'calculate':
                x1s = [np.array([0.5, 8, 23, 39])]
                fm = local_opt(x1s, train, test,
                            lower_bounds = [0.01, 7, 20, 38.5], upper_bounds = [5, 11, 31, 40],
                            great_threshold = 13, response_type = 'Wang',
                            phase_list = ['beginning of flowering'],
                            method='Nelder-Mead',
                            thresholds = [20], growing_period_length=170,
                            test_size = 0.02, maxiter=200)
                GDD_params = fm[0][0]
                print(GDD_params)
                model = initialise_as_GDD(model, *GDD_params, n_channels=5)
            else:
                GDD_params = GDD_init[fold]#[0.31621224, 7.37066436, 22.88481424, 39.40562477]
                print(GDD_params)
                model = initialise_as_GDD(model, *GDD_params, n_channels=5)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        if loss == 'NLL':
            criterion = nn.NLLLoss()
        elif loss == 'BCE':
            criterion = nn.BCELoss()
        elif loss == 'MSE':
            criterion = custom_loss()
        # Train the model on the current fold
        if freeze_params:
            for param in [model.u0, model.u1, model.u2, model.u3, model.p0]:
                param.requires_grad = False

        model_loss = fit_for_kf(epochs, model, criterion, optimizer, train_dl, val_dl, save_name = savename_prefix + str(fold + 1), CNN=CNN, bce=bce, verbose=verbose)
        loss_values.append(model_loss)

        #Now look at stats for model:
        model_dir = 'C:\\Users\\wlwc1989\\Documents\\Phenology_Test_Notebooks\\ML_algorithms\\saved_models\\'
        model_path = os.path.join(model_dir, savename_prefix + str(fold + 1) + ".pt")
        checkpoint = torch.load(model_path, weights_only=True)
        TS_model = model_class(*model_args).to(device)
        TS_model.load_state_dict(checkpoint['model_state_dict'])

        comparison_frame, logs = get_comparison_frame(savename_prefix + str(fold + 1), TS_model, val_dl, bce=bce, CNN=CNN, MMD=MMD, n_channels = n_channels, DTF = DTF, obs_method = obs_method)
        comparison_frame['Error'] = comparison_frame['observed'] - comparison_frame['fitted']

        bias_model = comparison_frame['Error'].mean()
        RMSE = np.sqrt(np.mean((comparison_frame['Error'] - bias_model)**2))
        STD = comparison_frame['fitted'].std() 
        Corr = comparison_frame['observed'].corr(comparison_frame['fitted'])

        RMSEs.append(RMSE)
        STDs.append(STD)
        Corrs.append(Corr)
        
        Mins.append(comparison_frame['Error'].min())
        LQs.append(comparison_frame['Error'].quantile(0.25))
        Medians.append(comparison_frame['Error'].median())
        UQs.append(comparison_frame['Error'].quantile(0.75))
        Maxs.append(comparison_frame['Error'].max())

        rsquared = r2_score(comparison_frame['observed'], comparison_frame['fitted'])
        bias = comparison_frame['observed'].mean() - comparison_frame['fitted'].mean()
        variance_modelled = comparison_frame[f'fitted'].var()
        print(rsquared)
        R2s.append(rsquared)
        Biases.append(bias)
        Variances.append(variance_modelled)
        if fold == 0:
            comparison_frame_full = comparison_frame
        else:
            comparison_frame_full = pd.concat((comparison_frame, comparison_frame_full))

    return comparison_frame_full, loss_values, R2s, Biases, Variances, RMSEs, STDs, Corrs, Mins, LQs, Medians, UQs, Maxs

def combined_comparison_frame(savenames, TS_models, dls, bces=[False], CNNs=[False], MMDs=[False], n_channels_list = [4], title = 'l'):
    multi_model_logs = []
    multi_model_stations = []
    multi_model_observed = []
    multi_model_years = []
    for model_ind, savename in enumerate(savenames):
        dl = dls[model_ind]
        TS_model = TS_models[model_ind]
        CNN = CNNs[model_ind]
        MMD = MMDs[model_ind]
        bce = bces[model_ind]
        n_channels = n_channels_list[model_ind]
        model_dir = 'C:\\Users\\wlwc1989\\Documents\\Phenology_Test_Notebooks\\ML_algorithms\\saved_models\\'
        model_path = os.path.join(model_dir, savename + ".pt")
        checkpoint = torch.load(model_path, weights_only=True)
        TS_model.load_state_dict(checkpoint['model_state_dict'])
        list_logs = []
        list_observed = []
        list_stations = []
        list_years = []
        for xb, yb in dl:
            with torch.no_grad():
                if CNN:
                    if MMD:
                        list_logs.append(TS_model(xb)[0])
                    else:
                        list_logs.append(TS_model(xb))
                else:
                    list_logs.append(TS_model(xb.transpose(1, 2)))
                list_years.append(xb[:, n_channels + 1, 0])
                list_stations.append(xb[:, n_channels, 0])
                #print(TS_model(xb.transpose(1, 2)))
            list_observed.append(yb)
        logs = torch.squeeze(torch.cat(list_logs))
        stations = torch.squeeze(torch.cat(list_stations))
        years = torch.squeeze(torch.cat(list_years))
        observed = torch.cat(list_observed)

        multi_model_logs.append(logs)
        multi_model_stations.append(stations)
        multi_model_observed.append(observed)
        multi_model_years.append(years)
    #multi_model_logs = torch.stack(multi_model_logs)
    #multi_model_stations = torch.stack(multi_model_stations)
    #multi_model_observed = torch.stack(multi_model_observed)
    #multi_model_years = torch.stack(multi_model_years)

    for model_ind, fitted in enumerate(multi_model_logs):
        Coords = {'Stations_id': (['statyear'], multi_model_stations[model_ind].numpy()),
                'year': (['statyear'], multi_model_years[model_ind].numpy())}
        Data = {f'fitted model {model_ind + 1}': (['statyear', 'time'], multi_model_logs[model_ind].numpy()),
                  f'observed model {model_ind + 1}': (['statyear', 'time'], multi_model_observed[model_ind].numpy())}
        da = xr.Dataset(data_vars = Data, coords = Coords)
        #da = da.set_coords('year').set_coords('Stations_id')
        #da = da.assign_coords({'statyear': 'Stations_id'})#station=('statyear', da['Stations_id'].data)).assign_coords(year=('statyear', da['year'].data))# {'statyear':'Stations_id', 'statyear':'year'})#.set_coords('year')
        da = da.set_xindex(['Stations_id', 'year']).sortby(['Stations_id', 'year'])#.reset_index(['Stations_id', 'year'])#set_xindex(['Stations_id', 'year'])
        #print(da)#.get_index())
        if model_ind == 0:
            full_da = da
        else:
            full_da = xr.combine_by_coords([da, full_da])#, dim='model')
    #full_da = full_da.set_coords('Stations_id').set_coords('year')
    return full_da#, multi_model_stations, multi_model_years

def get_comparison_frame(savename, TS_model, dl, bce=False, CNN=False, MMD=False, n_channels = 4, DTF = False, obs_method = False, prediction_day = 0):
    model_dir = 'C:\\Users\\wlwc1989\\Documents\\Phenology_Test_Notebooks\\ML_algorithms\\saved_models\\'
    model_path = os.path.join(model_dir, savename + ".pt")
    checkpoint = torch.load(model_path, weights_only=True)
    TS_model.load_state_dict(checkpoint['model_state_dict'])
    T_min = 9 + 10*(torch.tanh(TS_model.u1)) - 4 #normally multiplier = 2
    T_opt = 28 + 7*(torch.tanh(TS_model.u2)) - 4 #normally multiplier = 3
    T_max = 39 + 3.5*torch.tanh(TS_model.u3) - 2 #normally no multiplier
    #print(f'T_min = {T_min},\nT_opt = {T_opt},\nT_max = {T_max}')
    list_logs = []
    list_observed = []
    list_stations = []
    list_years = []
    for xb, yb in dl:
        with torch.no_grad():
            if CNN:
                if MMD:
                    list_logs.append(TS_model(xb)[0])
                else:
                    list_logs.append(TS_model(xb))
                    #print(TS_model(xb)[0, :])
            else:
                list_logs.append(TS_model(xb.transpose(1, 2)))
            list_years.append(xb[:, n_channels, 0])
            list_stations.append(xb[:, n_channels + 1, 0])
            #print(TS_model(xb.transpose(1, 2)))
        list_observed.append(yb)
    logs = torch.squeeze(torch.cat(list_logs))
    #print(logs[0, :])
    stations = torch.squeeze(torch.cat(list_stations))
    years = torch.squeeze(torch.cat(list_years))
    observed = torch.cat(list_observed)
    #print(logs.shape, torch.cat(list_logs, dim=0).shape)
    if not(DTF):
        if bce:
            fitted = torch.round(logs)
        else:
            fitted = torch.argmax(logs, dim=2)
    if DTF:
        logs = torch.round(logs)
        observed_days = mindex_at_0(observed)
        #observed_days = observed[:, 0]
        #observed_days = 90 + 100 - observed[:, 90]
        #observed_days = prediction_day + observed[:, prediction_day]
        if obs_method:
            fitted_days = observed_days + logs[range(0, logs.shape[0]), observed_days.int()]
        else:
            fitted_days = mindex_at_0(logs)
            #fitted_days = 90 + 100 - logs[:, 90]
            #fitted_days = logs[:, 90]
            #fitted_days = prediction_day + logs[:, prediction_day]
        print(torch.sum(fitted_days < 30), ' fits too small')
    else:
        if len(fitted.shape) == 1:
            fitted_days = len(fitted) - fitted.sum()
            observed_days = len(observed) - observed.sum()
        else:
            L = fitted.shape[1]
            fitted_days = L - fitted.sum(dim=1)
            observed_days = L - observed.sum(dim=1)
    
    #comparison_frame = pd.DataFrame({'fitted': fitted_days.numpy().squeeze(), 'observed': observed_days.numpy().squeeze()})
    if len(fitted.shape) > 1:
        comparison_frame = pd.DataFrame({'fitted': fitted_days.numpy().squeeze(), 'observed': observed_days.numpy().squeeze()})
    else:
        comparison_frame = pd.DataFrame({'fitted': [fitted_days.numpy().squeeze()], 'observed': [observed_days.numpy().squeeze()]})
        
    #print(stations.numpy().squeeze())
    #comparison_frame = pd.DataFrame({'Stations_id': stations.numpy().squeeze(), 'year': years.numpy().squeeze(), 'fitted': fitted_days.numpy().squeeze(), 'observed': observed_days.numpy().squeeze()})
    return comparison_frame, logs
    
class MMDLoss(nn.Module):
    def __init__(self, weight):
        super(MMDLoss, self).__init__()
        self.weight = weight

    def forward(self, source_features, target_features):
        # Compute the loss
        #loss = self.weight * torch.mean((((torch.mean(source_features, dim=0) - torch.mean(target_features, dim=0)) ** 2)) / torch.mean(torch.mean(source_features, dim=0)**2))
        source_mean = torch.mean(source_features, dim=0)
        target_mean = torch.mean(target_features, dim=0)
        source_std = torch.std(source_features, dim=0)
        out_of_region = 1 - (target_mean < source_mean + source_std).float()*(target_mean > source_mean - source_std).float()
        #loss = self.weight * torch.mean(torch.maximum(target_mean - (source_mean + source_std), torch.Tensor([0])) + torch.maximum(source_mean - source_std - target_mean, torch.Tensor([0]))) #torch.mean(out_of_region * torch.minimum(((1 - target_mean/(source_mean + source_std) )** 2), ((1 - target_mean/(source_mean - source_std) )** 2)))
        #loss = self.weight * torch.mean(out_of_region * torch.minimum(((1 - target_mean/(source_mean + source_std) )** 2), ((1 - target_mean/(source_mean - source_std) )** 2)))
        loss = (self.weight * (torch.mean(torch.log(target_features)) - torch.mean(torch.log(1 - source_features))))
        #print(loss)
        #loss = self.weight * torch.mean(torch.minimum(((1 - target_mean/source_mean) ** 2), torch.minimum(((1 - target_mean/(source_mean + source_std) )** 2), ((1 - target_mean/(source_mean - source_std) )** 2)))) 
        #loss = self.weight * torch.mean(((1 - torch.mean(target_features, dim=0)/torch.mean(source_features, dim=0)) ** 2))
        #loss = self.weight * torch.abs(((1 - torch.mean(target_features)/torch.mean(source_features) ** 2)))
        #print(loss)
        #print(torch.mean(source_features, dim=0).shape)
        return loss
class survival_loss(nn.Module):
    def __init__(self):
        super(survival_loss, self).__init__()
    def forward(self, logits, targets):
        #logits = torch.clamp(logits, min=1e-7, max=1 - 1e-7)
        #print(logits.shape)
        today_ind = targets.squeeze().to(int)
        veclen = len(today_ind)
        logits_at_day = logits[range(veclen), today_ind]
        logits_at_yesterday = logits[range(veclen), today_ind - 1]
        loss = -torch.sum(torch.log(logits_at_day - logits_at_yesterday))
        return loss
    
class problist_square_loss(nn.Module):
    def __init__(self):
        super(problist_square_loss, self).__init__()
    def forward(self, logits, targets):
        discrepancy = (targets - logits)**2
        #print(discrepancy[0, :])
        #print((discrepancy.sum(dim=1)**2)[0])
        if len(discrepancy.shape) == 1:
            return (discrepancy**2).mean()
        else:
            return (discrepancy.sum(dim=1)**2).mean()
        
class weighted_MSELoss(nn.Module):
    def __init__(self, weights = torch.Tensor(1 - 0.5*((np.arange(0, 1.63, 0.01) - 1.2)**2))):
        super().__init__()
        self.weights = weights
    def forward(self,inputs,targets):
        return torch.mean(((inputs - targets)**2 ) * self.weights)
def CausalConv1d(in_channels, out_channels, kernel_size, dilation=1, **kwargs):
   pad = (kernel_size - 1) * dilation
   return nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation, **kwargs)

class Causal_CNN_Classifier_KG(nn.Module):#
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, KG = True, MMD=False, target_features = torch.Tensor([0]), regression=False, KG_vpd = False, normalise_at_end = False,
                 T_min = 5, T_opt = 24, T_max = 39):
        super(Causal_CNN_Classifier_KG, self).__init__()
        self.normalise_at_end = normalise_at_end
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.KG = KG
        self.KG_vpd = KG_vpd
        self.regression=regression
        if MMD:
            self.target_features = target_features[:, :self.input_dim, :]
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.starting_mask = torch.nn.Parameter(torch.tensor([1] + [0 for count in range(self.input_dim - 1)]).float())
        self.MMD = MMD

        self.activation = nn.LeakyReLU(negative_slope=0.01)

        #self.wavenet_block1 = [CausalConv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=2**n) for n in range(1, 5)]
        #self.wavenet_block2 = [CausalConv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=2**n) for n in range(5)]
        #self.wavenet_block3 = [CausalConv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=2**n) for n in range(5)]

        self.conv1 = CausalConv1d(self.input_dim, hidden_dim, kernel_size=2, dilation=1)#stride=3,
        self.conv2 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=2)
        self.conv3 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=4)#stride=3, 
        self.conv4 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=8)
        self.conv5 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=16)
        self.conv6 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=32)
        self.conv7 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=64)
        self.conv7b = CausalConv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=128)
        self.conv8 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=1)#stride=3,
        self.conv9 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=2)
        self.conv10 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=4)#stride=3, 
        self.conv11 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=8)
        self.conv12 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=16)
        self.conv13 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=32)
        self.conv14 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=64)
        self.conv14b = CausalConv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=128)
        self.conv15 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=1)#stride=3,
        self.conv16 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=2)
        self.conv17 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=4)#stride=3, 
        self.conv18 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=8)
        self.conv19 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=16)
        self.conv20 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=32)
        self.conv21 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=64)
        self.conv21b = CausalConv1d(hidden_dim, hidden_dim, kernel_size=2, dilation=128)
        self.wave1 = [self.conv1, self.conv2, self.conv3, self.conv4, 
                                  self.conv5, self.conv6, self.conv7, self.conv7b]
        self.wave2 = [self.conv8, 
                      self.conv9, self.conv10, self.conv11, self.conv12, 
                      self.conv13, self.conv14, self.conv14b]
        self.wave3 = [self.conv15, self.conv16,
                      self.conv17, self.conv18, self.conv19, 
                      self.conv20, self.conv21, self.conv21b]
        if self.MMD:
            if self.num_layers == 1:
                self.full_waves = []
            if self.num_layers == 2:
                self.full_waves = self.wave2
            if self.num_layers == 3:
                self.full_waves = self.wave2 + self.wave3
        else:
            if self.num_layers == 1:
                self.full_waves = self.wave1
            if self.num_layers == 2:
                self.full_waves = self.wave1 + self.wave2
            if self.num_layers == 3:
                self.full_waves = self.wave1 + self.wave2 + self.wave3

        self.fc = nn.Linear(hidden_dim, output_dim)  # Fully connected layer for classification
        
        #self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False)

        self.sig = nn.Sigmoid()

        #u1, u2, u3 = cardinal_temps_to_ML_params(T_min, T_opt, T_max)
        self.u0 = torch.nn.Parameter(torch.Tensor([1]))
        self.u1 = torch.nn.Parameter(torch.Tensor([0]))
        self.u2 = torch.nn.Parameter(torch.Tensor([0]))
        self.u3 = torch.nn.Parameter(torch.Tensor([0]))
        #self.u0 = torch.nn.Parameter(torch.Tensor([1]))
        #self.u1 = torch.nn.Parameter(torch.Tensor([u1]))
        #self.u2 = torch.nn.Parameter(torch.Tensor([u2]))
        #self.u3 = torch.nn.Parameter(torch.Tensor([u3]))
        
        self.fc_u0_1 = nn.Linear(2, 2)
        self.fc_u0_2 = nn.Linear(2, 2)
        self.fc_u0_3 = nn.Linear(2, 1)
        self.fc_u1_1 = nn.Linear(2, 2)
        self.fc_u1_2 = nn.Linear(2, 2)
        self.fc_u1_3 = nn.Linear(2, 1)
        self.fc_u2_1 = nn.Linear(2, 2)
        self.fc_u2_2 = nn.Linear(2, 2)
        self.fc_u2_3 = nn.Linear(2, 1)


        self.p0 = torch.nn.Parameter(torch.Tensor([13]))
        

    def forward(self, x0):
        #print(x0.shape)
        x = x0[:, :self.input_dim, :].clone()
        x = x*self.starting_mask[None, :, None]
        #print(x[0, :, 0], x[0, :, 10], x[0, :, 20])
        #print(x.shape)
        if self.KG_vpd:
            x_VPD = x[:, np.array([2, 3]), :].clone()
        #print(x_VPD.shape)
        if self.MMD:
            target_x = self.target_features
        #print(x.max())
        #print(x0.shape)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device) # Initialize hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device) # Initialize cell state
        # Apply Wang Engel
        if self.KG_vpd:
            x_for_temps = []
            for fc_layers in [[self.fc_u0_1, self.fc_u0_2, self.fc_u0_3], [self.fc_u1_1, self.fc_u1_2, self.fc_u1_3], [self.fc_u2_1, self.fc_u2_2, self.fc_u2_3]]:
                x_for_cardinal_temp = torch.swapaxes(fc_layers[0](torch.swapaxes(x_VPD, 2, 1)), 2, 1)
                x_for_cardinal_temp = self.activation(x_for_cardinal_temp)
                x_for_cardinal_temp = torch.swapaxes(fc_layers[1](torch.swapaxes(x_for_cardinal_temp, 2, 1)), 2, 1)
                x_for_cardinal_temp = self.activation(x_for_cardinal_temp)
                x_for_cardinal_temp = torch.swapaxes(fc_layers[2](torch.swapaxes(x_for_cardinal_temp, 2, 1)), 2, 1)
                x_for_temps.append(x_for_cardinal_temp)
            T_min = 9 + 10*(torch.tanh(x_for_temps[0])).squeeze() - 4 #normally multiplier = 2
            T_opt = 28 + 7*(torch.tanh(x_for_temps[1])).squeeze() - 4 #normally multiplier = 3
            T_max = 39 + 3.5*torch.tanh(x_for_temps[2]).squeeze()#normally no multiplier
            #print(T_min.shape)

        else:
            T_min = 9 + 10*(torch.tanh(self.u1)) - 4 #normally multiplier = 2
            T_opt = 28 + 7*(torch.tanh(self.u2)) - 4 #normally multiplier = 3
            T_max = 39 + 3.5*torch.tanh(self.u3)#normally no multiplier
        alpha = np.log(2)/torch.log( (T_max - T_min)/(T_opt - T_min) )
        #print(T_min, T_opt, T_max, alpha)
        #print(((T_opt - T_min)**(2*alpha)))
        #print((((x[:, 0, :] - T_min)*(x[:, 0, :] >= T_min))**(2*alpha))[0, :])
        #print(((2*(((x[:, 0, :] - T_min)*(x[:, 0, :] >= T_min))**alpha))*((T_opt - T_min)**alpha))[0, :])
        #print((self.u0 * (x[:, 0, :] <= T_max) * ( (2*(((x[:, 0, :] - T_min)*(x[:, 0, :] >= T_min))**alpha))*((T_opt - T_min)**alpha) - (((x[:, 0, :] - T_min)*(x[:, 0, :] >= T_min))**(2*alpha)) ) / ((T_opt - T_min)**(2*alpha)))[0, :])
        #p = self.u0 * (x[:, 0, :] <= T_max) * ( (2*(((x[:, 0, :] - T_min)*(x[:, 0, :] >= T_min)*(x[:, 0, :] <= T_max))**alpha))*((T_opt - T_min)**alpha) - (((x[:, 0, :] - T_min)*(x[:, 0, :] >= T_min)*(x[:, 0, :] <= T_max))**(2*alpha)) ) / ((T_opt - T_min)**(2*alpha))
        #p = p.unsqueeze(1)
        #if torch.any(torch.isnan(p[:, 0, :])):
        #    print(x[:, 0, :][torch.isnan(p[:, 0, :])])
        #    print('aha!')
        #if not(alpha > 0):
        #    print(alpha, T_min, T_opt, T_max)
        beta = 1
        #print(T_min, T_opt, T_max)
        #print(alpha)
        #print(((2*(x - T_min)*(x >= T_min))**alpha))
        #print(x.max())
        if self.KG:
            #x[:, 0, :] = ((2*(x[:, 0, :] - T_min)*(x[:, 0, :] >= T_min))**alpha)*((T_opt - T_min)**alpha) - (((x[:, 0, :] - T_min)*(x[:, 0, :] >= T_min))**(2*alpha))
            #print(alpha[0, :], (T_opt - T_min)[0, :])
            #t = (x[:, 0, :] - T_min)*torch.sign(x[:, 0, :] - T_min)
            #print(t.min())
            #s =  (2*(t.pow(alpha))*(x[:, 0, :] > T_min))
            #r = (s*((T_opt - T_min).pow(alpha)))
            #p = self.u0 * (x[:, 0, :] <= T_max) * r
            #q = p - self.u0 * (((x[:, 0, :] - T_min)*(x[:, 0, :] > T_min)).pow(2*alpha)) 
            #x[:, 0, :] = q / ((T_opt - T_min).pow(2*alpha))
            x[:, 0, :] = self.u0 * (x[:, 0, :] <= T_max) * ( (2*(((x[:, 0, :] - T_min)*(x[:, 0, :] >= T_min) + (x[:, 0, :] <= T_min).float()).pow(alpha))*(x[:, 0, :] >= T_min))*((T_opt - T_min).pow(alpha)) - ((((x[:, 0, :] - T_min)*(x[:, 0, :] >= T_min) + (x[:, 0, :] <= T_min).float()).pow(2*alpha))*(x[:, 0, :] >= T_min)) ) / ((T_opt - T_min).pow(2*alpha))
            if torch.any(torch.isnan(x[:, 0, :])):
                print(T_min, T_opt, alpha)
        #print(x[0, 0, :])
        if self.input_dim >= 3:
            x[:, 1, :] = 0.5*(1 + torch.tanh(2*(x[:, 1, :] - self.p0)))
        #print(x)
        #print(x.shape)
        if self.MMD:
            for i, conv in enumerate(self.wave1):
                x = conv(x)/self.hidden_dim
                x = x[:, :, :-conv.padding[0]]
                x = self.activation(x)

                target_x = conv(target_x)
                target_x = target_x[:, :, :-conv.padding[0]]
                target_x = self.activation(target_x)
            transferred_x = torch.swapaxes(self.fc2(torch.swapaxes(x, 2, 1)), 1, 2)
            transferred_x = self.activation(transferred_x)

            target_x = torch.swapaxes(self.fc2(torch.swapaxes(target_x, 2, 1)), 1, 2)
            target_x = self.activation(target_x)
            
            transferred_x = self.sig(transferred_x)
            target_x = self.sig(target_x)

        for i, conv in enumerate(self.full_waves):
            x = conv(x)/self.hidden_dim
            x = x[:, :, :-conv.padding[0]]
            x = self.activation(x)

        
        # Classify all layers using fully connected layer
        out_space = torch.swapaxes(self.fc(torch.swapaxes(x, 2, 1)), 1, 2)*self.hidden_dim # (batch, output_dim)
        #print(out_space)
        #print(out_space.shape)
        if self.output_dim == 1:
            if self.regression:
                out_scores = out_space[:, :, [162]]
            else:
                if self.normalise_at_end:
                    #print(out_space.shape)
                    out_space = torch.cumsum(torch.abs(out_space), dim=2) - 20
                out_scores = self.sig(out_space - 20)
                #print(out_scores.max(), out_scores.min())
                #print(out_scores[0, :, :])
        else:
            out_scores = F.log_softmax(out_space, dim=2)
        #print(out_scores.max())
        if self.MMD:
            return out_scores, transferred_x, target_x
        else:
            return out_scores
        
def cardinal_temps_to_ML_params(T_min, T_opt, T_max):
    u1 = np.arctanh((T_min - 5)/10)
    u2 = np.arctanh((T_opt - 24)/7)
    u3 = np.arctanh((T_max - 39)/3.5)
    return u1, u2, u3

def initialise_as_GDD(model, scale, T_min, T_opt, T_max, n_channels = 1):
    for i, conv in enumerate(model.wave1):
        #print(conv.weight, conv.bias)
        #if i == 1 and n_channels != 1:
        #    print(conv.weight.shape, conv.bias.shape)
        #    weight_temp = torch.ones((2, model.hidden_dim, 1))
        #    weight_else = torch.zeros(2, model.hidden_dim, n_channels - 1)
        #    conv.weight = nn.Parameter(torch.concat((weight_temp, weight_else), dim=2).float().transpose(0, 1))
        #    print(conv.weight.shape, conv.bias.shape)
        #else:
        nn.init.ones_(conv.weight)
        nn.init.zeros_(conv.bias)
        #print(conv.weight, conv.bias)
    for conv in model.wave2:
        #print(conv.weight, conv.bias)
        #print(conv.weight.shape, conv.bias.shape)
        conv.weight = nn.Parameter(torch.tensor([[[0, 1] for count in range(model.hidden_dim)] for count in range(model.hidden_dim)]).float().transpose(0, 1))
        nn.init.zeros_(conv.bias)
        #print(conv.weight.shape, conv.bias.shape)
        #print(conv.weight, conv.bias)
    for conv in model.wave3:
        conv.weight = nn.Parameter(torch.tensor([[[0, 1] for count in range(model.hidden_dim)] for count in range(model.hidden_dim)]).float().transpose(0, 1))
        nn.init.zeros_(conv.bias)
    nn.init.eye_(model.fc.weight)
    nn.init.zeros_(model.fc.bias)
    model.u0 = nn.Parameter(torch.tensor([scale]).float())
    u1, u2, u3 = cardinal_temps_to_ML_params(T_min, T_opt, T_max)
    model.u1 = nn.Parameter(torch.tensor([u1]))#.float())
    model.u2 = nn.Parameter(torch.tensor([u2]))#.float())
    model.u3 = nn.Parameter(torch.tensor([u3]))#.float())
    return model

def forward_look(self, x0):
    #print(x0.shape)
    x = x0[:, :self.input_dim, :].clone()
    T_min = 9 + 10*(torch.tanh(self.u1)) - 4 #normally multiplier = 2
    T_opt = 28 + 7*(torch.tanh(self.u2)) - 4 #normally multiplier = 3
    T_max = 39 + 3.5*torch.tanh(self.u3) - 2 #normally no multiplier
    alpha = np.log(2)/torch.log( (T_max - T_min)/(T_opt - T_min) )
    x[:, 0, :] = self.u0 * (x[:, 0, :] <= T_max) * ( (2*(((x[:, 0, :] - T_min)*(x[:, 0, :] >= T_min))**alpha))*((T_opt - T_min)**alpha) - (((x[:, 0, :] - T_min)*(x[:, 0, :] >= T_min))**(2*alpha)) ) / ((T_opt - T_min)**(2*alpha))
    if torch.any(torch.isnan(x[:, 0, :])):
        print(T_min, T_opt, alpha)
    if self.input_dim >= 3:
        x[:, 1, :] = 0.5*(1 + torch.tanh(2*(x[:, 1, :] - self.p0)))
    for i, conv in enumerate(self.full_waves):
        x = conv(x)
        x = x[:, :, :-conv.padding[0]]
        x = self.activation(x)
    # Classify all layers using fully connected layer
    out_space = torch.swapaxes(self.fc(torch.swapaxes(x, 2, 1)), 1, 2) # (batch, output_dim)
    #print(out_space.shape)
    if self.output_dim == 1:
        if self.regression:
            out_scores = out_space[:, :, [162]]
    return out_space
def loss_batch(model, loss_func, xb, yb, opt=None, CNN=False, bce=False, MMD=False, MMD_loss = None):
    if CNN:
        if MMD:
            outputs, transferred_x, target_x = model(xb)
            loss = loss_func(torch.squeeze(outputs.transpose(1, 2)), torch.squeeze(yb.float())) + MMD_loss(transferred_x, target_x)
        else:
            outputs = model(xb)
            #print(torch.squeeze(outputs.transpose(1, 2)).shape, torch.squeeze(yb.float()).shape, (torch.squeeze(outputs.transpose(1, 2)) - torch.squeeze(yb.float())))
            loss = loss_func(torch.squeeze(outputs.transpose(1, 2)), torch.squeeze(yb.float()))
    elif bce:
        outputs = model(xb.transpose(1, 2))
        #print(outputs.shape)
        loss = loss_func(torch.squeeze(outputs), torch.squeeze(yb.float()))#.transpose(1, 2)
    else:
        outputs = model(xb.transpose(1, 2))
        loss = loss_func(outputs.transpose(1, 2), yb.long())

    if opt is not None:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
        opt.step()
        opt.zero_grad()
    #print(loss.item())
    return loss.item(), len(xb)

def fit_for_kf(epochs, model, loss_func, opt, train_dl, valid_dl, save_name = 'best_model', plot_opt = False, CNN=False, bce=False, verbose=False):
    # Variables to store training history
    train_losses = []
    val_losses = []
    best_loss = 100000
    best_epoch = 0

    for epoch in range(epochs):
        running_loss = 0.0
        running_samples = 0
        model.train()

        for xb, yb in train_dl:
            batch_loss, batch_len = loss_batch(model, loss_func, xb, yb, opt, CNN=CNN, bce=bce)
            running_loss += batch_loss*batch_len
            running_samples += batch_len

        train_loss = running_loss/running_samples
        
        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb, CNN=CNN, bce=bce) for xb, yb in valid_dl]
            )
            
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        # Save the best model (based on validation accuracy)
        if train_loss < best_loss:
            best_loss = train_loss
            best_epoch = epoch + 1
            best_model_state = deepcopy(model.state_dict())
            model_loss = val_loss

        #Save the model at the last epoch
        if epoch == epochs - 1:
            model_dir = 'C:\\Users\\wlwc1989\\Documents\\Phenology_Test_Notebooks\\ML_algorithms\\saved_models\\'
            model_path = os.path.join(model_dir, save_name + ".pt")
            torch.save({'epoch': best_epoch, 'model_state_dict': best_model_state}, model_path)
        
        if verbose and epoch % 50 == 0:
            print(epoch, train_loss, val_loss)
    #print(f'Loss: {best_loss}')
    if plot_opt:
        plot_train_val_loss(epochs, train_losses, val_losses, best_epoch)
    return model_loss

def fit(epochs, model, loss_func, opt, train_dl, valid_dl, save_name = 'best_model', CNN = False, bce=False, MMD = False, MMD_loss = None):
    # Variables to store training history
    train_losses = []
    val_losses = []
    best_loss = 50000
    best_epoch = 0

    for epoch in range(epochs):
        running_loss = 0.0
        running_samples = 0
        model.train()
        running_var = 0
        running_var_val = 0

        for xb, yb in train_dl:
            batch_loss, batch_len = loss_batch(model, loss_func, xb, yb, opt, CNN=CNN, bce=bce, MMD = MMD, MMD_loss = MMD_loss)
            running_loss += batch_loss*batch_len
            running_samples += batch_len

        train_loss = running_loss/running_samples
        
        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb, CNN=CNN, bce=bce, MMD = MMD, MMD_loss = MMD_loss) for xb, yb in valid_dl]
            )
            
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        # Save the best model (based on validation accuracy)
        if train_loss < best_loss:
            best_loss = train_loss
            best_epoch = epoch + 1
            best_model_state = deepcopy(model.state_dict())

        #Save the model at the last epoch
        if epoch == epochs - 1:
            model_dir = 'C:\\Users\\wlwc1989\\Documents\\Phenology_Test_Notebooks\\ML_algorithms\\saved_models\\'
            model_path = os.path.join(model_dir, save_name + ".pt")
            torch.save({'epoch': best_epoch, 'model_state_dict': best_model_state}, model_path)
        if epoch % 5 == 0:
            print(epoch, train_loss, val_loss)
    plot_train_val_loss(epochs, train_losses, val_losses, best_epoch, max_y = 2000)


def split_ds_by_AEZ(ds):
    ds_semiarid = ds.loc[ds['AEZ'].isin([1, 4, 10])]
    ds_humid = ds.loc[ds['AEZ'].isin([3])]
    ds_subhumid = ds.loc[ds['AEZ'].isin([2, 5, 11, 14])]
    ds_arid = ds.loc[ds['AEZ'].isin([26, 29])]
    ds_city = ds.loc[ds['AEZ'].isin([32])]
    print(f'\nnum arid: {len(ds_arid)}',
        f'\nnum city: {len(ds_city)}',
        f'\nnum semiarid: {len(ds_semiarid)}',
        f'\nnum subhumid: {len(ds_subhumid)}',
        f'\nnum humid: {len(ds_humid)}')
    ds_dict = {'arid': ds_arid,
               'city': ds_city,
               'semiarid': ds_semiarid,
               'subhumid': ds_subhumid,
               'humid': ds_humid}
    return ds_dict

def split_ds_by_AEZ3(ds):
    ds_arid_low = ds.loc[ds['AEZ'].isin([1, 26, 29])]
    ds_humid_low = ds.loc[ds['AEZ'].isin([2, 3, 32])]
    ds_high = ds.loc[ds['AEZ'].isin([4, 5, 10, 11, 14])]
    print(f'\nnum arid low: {len(ds_arid_low)}',
        f'\nnum humid low: {len(ds_humid_low)}',
        f'\nnum high: {len(ds_high)}')
    ds_dict = {'arid low': ds_arid_low,
               'humid low': ds_humid_low,
               'high or cool': ds_high}
    return ds_dict
def split_ds_by_AEZ4(ds):
    ds_arid_low = ds.loc[ds['AEZ'].isin([1, 26, 29])].groupby(['Stations_id', 'Referenzjahr']).head(2)
    ds_humid_low = ds.loc[ds['AEZ'].isin([2, 3])].groupby(['Stations_id', 'Referenzjahr']).head(2)
    ds_high = ds.loc[ds['AEZ'].isin([4, 5])].groupby(['Stations_id', 'Referenzjahr']).head(2)
    ds_cool = ds.loc[ds['AEZ'].isin([10, 11, 14])].groupby(['Stations_id', 'Referenzjahr']).head(2)
    ds_city = ds.loc[ds['AEZ'].isin([32])].groupby(['Stations_id', 'Referenzjahr']).head(2)
    print(f'\nnum arid low: {len(ds_arid_low)}',
        f'\nnum city: {len(ds_city)}',
        f'\nnum high: {len(ds_high)}',
        f'\nnum humid low: {len(ds_humid_low)}',
        f'\nnum cool: {len(ds_cool)}')
    ds_dict = {'arid low': ds_arid_low,
               'city': ds_city,
               'high': ds_high,
               'humid low': ds_humid_low,
               'cool': ds_cool}
    return ds_dict

def ML_tensor_from_ds(ds):
    ds=ds.rename(columns={'Referenzjahr':'year'})
    #print(len(ds))
    ds = ds.dropna(how='all')
    #print(len(ds))
    ds = ds.loc[ds['observed time to beginning of flowering'] < 140]
    skip = 1
    numsteps = int(163 // skip)
    new_series = []
    ds.loc[:, [f't2max at day {skip*n}' for n in range(numsteps)]] = ds.loc[:, [f't2max at day {skip*n}' for n in range(numsteps)]].values - ds.loc[:, [f't2m at day {skip*n}' for n in range(numsteps)]].values
    ds.loc[:, [f't2min at day {skip*n}' for n in range(numsteps)]] = ds.loc[:, [f't2m at day {skip*n}' for n in range(numsteps)]].values - ds.loc[:, [f't2min at day {skip*n}' for n in range(numsteps)]].values
    for variable_name in ['t2m', 't2max', 't2min', 'photoperiod', 'vpd']:#['t2m', 'photoperiod', 'vpd']: #, 'ssrd', 'tp', 't2max', 't2min'
        data = ds[[f'{variable_name} at day {skip*n}' for n in range(numsteps)]].values
        #if variable_name != 't2m' and variable_name != 'photoperiod':
        #    scaler = StandardScaler()
        #    scaler.fit(data)
        #    data = scaler.transform(data)
        new_variable_series = torch.Tensor(data)
        new_series.append(new_variable_series)

    day_series = torch.Tensor([[(skip*n)/10 for n in range(numsteps)] for count in range(len(ds))])
    year_series = torch.Tensor([ds['year'].values for count in range(numsteps)]).T
    id_series = torch.Tensor([ds['Stations_id'].values for count in range(numsteps)]).T
    pd.Categorical(ds['Management'], categories= ['Drought', 'Low N', 'Low pH', 'Maize Streak Virus', 'Optimal']).codes
    var_series = torch.Tensor([pd.Categorical(ds['vargroup'], categories= ['EIHY', 'EPOP', 'ILHY', 'ILPO']).codes for count in range(numsteps)]).T
    man_series = torch.Tensor([pd.Categorical(ds['Management'], categories= ['Drought', 'Low N', 'Low pH', 'Maize Streak Virus', 'Optimal']).codes for count in range(numsteps)]).T
    X_tensor = torch.swapaxes(torch.stack((*new_series, day_series, var_series, man_series, year_series, id_series)), 0, 1)
    ds.loc[:, [f'DTF at day {n}' for n in range(193)]] = np.tile(ds['observed time to beginning of flowering'].values, (193, 1)).T - np.tile(np.arange(0, 193), (len(ds),1))
    ds.loc[:, [f'dev stage at day {skip*n}' for n in range(numsteps)]] = (ds.loc[:, [f'DTF at day {skip*n}' for n in range(numsteps)]] < 0).astype(int).values
    #print(ds[[f'dev stage at day {skip*n}' for n in range(numsteps)]])
    y_tensor = torch.Tensor(ds[[f'dev stage at day {skip*n}' for n in range(numsteps)]].astype('int64').values) #torch.Tensor(ds_inputs_SSA['observed time to beginning of flowering'].astype('int64').values)
    #NDVI_labels_SSA = torch.Tensor(ds_inputs_SSA[[f'DTF at day {skip*n}' for n in range(numsteps)]].astype('int64').values) #torch.Tensor(ds_inputs_SSA['observed time to beginning of flowering'].astype('int64').values)
    #NDVI_labels_SSA = torch.Tensor(ds_inputs_SSA[[f'DTF at day {0}' for n in range(numsteps)]].astype('int64').values)
    return X_tensor, y_tensor


def K_fold_transfer_regions(ds_dict, transfer_method = 'KG'):
    if transfer_method == 'last_layer':
        lr = 0.01
        epochs = 500
    else:
        #lr = 0.05
        lr = 0.001
        epochs=100
        #epochs = 500
    #save_name = 'CCNN_KG_DE_pre_2022_for_TL3'
    save_name = 'CCNN_KG_DE_pre_2022_for_TL2'
    bs = 500
    n_channels = 5
    input_dim = n_channels # Example: 100-dimensional word embeddings
    hidden_dim = 4#8
    normalise_at_end = False#True
    num_layers = 3
    output_dim = 1 
    KG = True
    MMD = False
    regression = False
    KG_vpd = False
    CNN = True
    bce = True
    model_args = (input_dim, hidden_dim, num_layers, output_dim, KG, MMD, None, regression, KG_vpd, normalise_at_end, 8, 22, 39)
    # initialise lists for storing
    SSE_all_reg = []
    length_all_reg = []
    obsvars_all_reg = []
    regs = []
    comparison_frames = []
    for region in ds_dict.keys():
        #get data for specific region
        ds = ds_dict[region]
        if len(ds) < 20:
            continue
        regs.append(region)
        #convert to pytorch format
        #X_tensor, y_tensor = ML_tensor_from_ds(ds)
        #train_ds, val_ds = train_test_ds_from_tensors(X_tensor, y_tensor, train_proportion=1)
        #run K-fold cross validation
        criterion = problist_square_loss()
        comparison_frame, loss_values, R2s, Biases, Variances, RMSEs, SSEs, STDs, Corrs, Mins, LQs, Medians, UQs, Maxs, lengths, obsvars = K_fold_transfer(5, ds, Causal_CNN_Classifier_KG, 
                                                                                                                save_name, epochs, bs, model_args, 
                                                                                                                transfer_method=transfer_method, lr = lr, 
                                                                                                                loss = 'MSE', CNN=CNN, bce=bce, 
                                                                                                                custom_loss=criterion, DTF = False)
        comparison_frames.append(comparison_frame)
        print(region, R2s)
        #Store sum of squared errors, variance of obs and length of folds for evaluation
        SSE_all_reg.append(SSEs)
        length_all_reg.append(lengths)
        obsvars_all_reg.append(obsvars)
    results_dict = {'SSEs': SSE_all_reg,
                    'observed variances': obsvars_all_reg,
                    'fold lengths': length_all_reg,
                    'regions': regs,
                    'comparison_frames': comparison_frames}
    return results_dict

def skill_score_all_reg(score_dict):
    SSE_all_reg = np.array(score_dict['SSEs'])
    obsvars_all_reg = np.array(score_dict['observed variances'])
    length_all_reg = np.array(score_dict['fold lengths'])
    MSE_all_reg = SSE_all_reg.sum(axis=0)/length_all_reg.sum(axis = 0)
    R2_all_reg = 1 - (SSE_all_reg.sum(axis=0)/((length_all_reg - 1)*obsvars_all_reg).sum(axis=0))
    R2_all_reg_split = 1 - (SSE_all_reg/((length_all_reg - 1)*obsvars_all_reg))
    #R2_pooled = 1 - (SSE_all_reg.sum(axis=1)/((length_all_reg - 1)*obsvars_all_reg).sum(axis=1))
    R2_pooled = []
    for cf in score_dict['comparison_frames']:
        R2_pooled.append(r2_score(cf['observed'], cf['fitted']))
    R2_pooled = np.array(R2_pooled)
    return R2_all_reg, R2_all_reg_split, R2_pooled

def make_df_from_score_dict(score_dict, TL_method = 'NA', variety = 'NA'):
    R2s, R2s_split, R2s_pooled = skill_score_all_reg(score_dict)
    df = pd.DataFrame()
    df['R2 score'] = R2s_split.mean(axis=1)
    df_pooled = pd.DataFrame()
    df_pooled['R2 score'] = R2s_pooled
    df['Variety'] = variety
    df_pooled['Variety'] = variety
    df_full_model = pd.DataFrame()
    df_full_model['Variety'] = [variety]
    df_full_model['R2 score'] = [R2s.mean()]
    return df, df_pooled, df_full_model