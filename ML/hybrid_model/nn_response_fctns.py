import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from scipy.integrate import quad

import seaborn as sns
from scipy import stats
from scipy.stats import truncnorm
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

import torch
import torch.nn as nn
import torch.optim as optim
#import torchvision
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
#import torchinfo

import warnings
import os
from copy import deepcopy

import plotting
import dataset_fctns
import modelling_fctns
from ML_fctns import *

from datetime import datetime
from dateutil.relativedelta import relativedelta
#from suntimes import SunTimes  
import sys
sys.path.append("C:\\Users\\wlwc1989\\Documents\\Phenology_Test_Notebooks\\phenology_dwd\\optimisation_experiments")
from optimise_GDD_fctns import *

def run_GDD(x, ds, driver_variable, latlon_proj = True, response_type = 'Trapezoid', 
                             phase_list = ['beginning of flowering'], exclude_unfinished = False,
                             growing_period_length = 300, thresholds = [100], 
                             title_extra='', method='scatter', savename = False, plot=False, col='blue',
                             index_cols = ['lat', 'lon', 'Management', 'vargroup', 'Stations_id', 'year']): 
    #Change index_cols to ['year', 'Stations_id'] if using DE dataset
    len_index_cols = len(index_cols)
    if response_type == 'Trapezoid':
        def response(meantemp):
            #return x[0]*modelling_fctns.Wang_Engel_Temp_response(meantemp, x[1], x[2], x[3])
            return x[0]*modelling_fctns.Trapezoid_Temp_response(meantemp, x[1], x[2], 0.2, 3)#x[3], x[4])
    elif response_type == 'Wang':
        def response(meantemp):
            #return x[0]*modelling_fctns.Wang_Engel_Temp_response(meantemp, x[1], x[2], x[3])
            return x[0]*modelling_fctns.Wang_Engel_Temp_response(meantemp, x[1], x[2], x[3])
    elif response_type == 'Convolved':
        table = vec_expint(x[1], x[2], x[3], np.arange(0, 50, 0.5), 10, 3)#, x[4], x[5])#x[2]
        def response(meantemp):
            return x[0]*table[(np.round(meantemp/5, decimals = 1)*10).astype(int)]*(meantemp > 0)
    elif response_type == 'Convolved_vary_spread':
        table = vec_expint(x[1], x[2], x[3], np.arange(0, 50, 0.5), x[4], x[5])#, x[4], x[5])#x[2]
        def response(meantemp):
            return x[0]*table[(np.round(meantemp/5, decimals = 1)*10).astype(int)]*(meantemp > 0)
    elif response_type == 'multi_phase':
        def response(meantemp):
            #return x[0]*modelling_fctns.Wang_Engel_Temp_response(meantemp, x[1], x[2], x[3])
            return x[0]*modelling_fctns.Wang_Engel_Temp_response(meantemp, x[1], x[2], x[3])
        responses = [lambda meantemp: x[0]*modelling_fctns.Wang_Engel_Temp_response(meantemp, x[1], x[2], x[3]),
                     lambda meantemp: x[4]*modelling_fctns.Wang_Engel_Temp_response(meantemp, x[5], x[6], x[7])]
    driver_columns = [f'{driver_variable} at day {day}' for day in range(growing_period_length)]
    ds_for_model = ds[driver_columns + index_cols].copy()
    #ds_for_model.loc[:, driver_columns] = ds_for_model.loc[:, driver_columns]#.round(decimals = 10).astype(np.float64)
    if response_type == 'multi_phase':
        t_dev = np.zeros(len(ds_for_model))
        for i, colm in enumerate(driver_columns):
            resp = modelling_fctns.phase_dependent_response2(ds_for_model.loc[:, colm], t_dev, responses, thresholds)
            #resp2 = response(ds_for_model.loc[:, colm])
            #if np.any(resp != resp2):
            #    print(resp, resp2)
            #print(resp)
            t_dev += resp
            ds_for_model.loc[:, colm] = t_dev
    else:
        ds_for_model.loc[:, driver_columns] = response(ds_for_model[driver_columns]).cumsum(axis=1)
    model_dev_time_series = ds_for_model.values.T
    column_names = [np.array([f'modelled time to {phase}' for phase in phase_list]+ index_cols)]
    thresholds_for_phase = thresholds[-1:]
    #print(thresholds_for_phase, model_dev_time_series.shape[1])
    phase_dates_array = np.zeros((len(thresholds_for_phase), model_dev_time_series.shape[1]))#
    for obs_index in range(model_dev_time_series.shape[1]):
        #print(model_dev_time_series[:-2, obs_index])
        #print(model_dev_time_series[:-len_index_cols, obs_index])#.astype(np.float64))
        phase_dates_array[:, obs_index] = np.digitize(thresholds_for_phase, model_dev_time_series[:-len_index_cols, obs_index].astype(np.float64))    
    #print(phase_dates_array)
    phase_dates_array = np.concatenate([phase_dates_array, *[[model_dev_time_series[-(k + 1)]] for k in reversed(range(len_index_cols))]], axis=0)
    #print(column_names)
    phase_dates_array = pd.DataFrame(phase_dates_array.T, columns = column_names[0])
    #print(ds.columns.to_list(), phase_dates_array.columns.to_list())
    comparison_array = ds.T.drop_duplicates().T.merge(phase_dates_array, how='left', on=index_cols).dropna(how='all')
    if plot:
        plot_from_comparison_array(comparison_array, title_extra=title_extra, method=method, savename=savename, 
                                  phase_list=phase_list, exclude_unfinished=exclude_unfinished, growing_period_length=growing_period_length,
                                  col=col)
    return comparison_array

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
    X_tensor = torch.swapaxes(torch.stack((*new_series, day_series, year_series, id_series)), 0, 1)
    ds.loc[:, [f'DTF at day {n}' for n in range(193)]] = np.tile(ds['observed time to beginning of flowering'].values, (193, 1)).T - np.tile(np.arange(0, 193), (len(ds),1))
    ds.loc[:, [f'dev stage at day {skip*n}' for n in range(numsteps)]] = (ds.loc[:, [f'DTF at day {skip*n}' for n in range(numsteps)]] < 0).astype(int).values
    #print(ds[[f'dev stage at day {skip*n}' for n in range(numsteps)]])
    y_tensor = torch.Tensor(ds[[f'dev stage at day {skip*n}' for n in range(numsteps)]].astype('int64').values) #torch.Tensor(ds_inputs_SSA['observed time to beginning of flowering'].astype('int64').values)
    #NDVI_labels_SSA = torch.Tensor(ds_inputs_SSA[[f'DTF at day {skip*n}' for n in range(numsteps)]].astype('int64').values) #torch.Tensor(ds_inputs_SSA['observed time to beginning of flowering'].astype('int64').values)
    #NDVI_labels_SSA = torch.Tensor(ds_inputs_SSA[[f'DTF at day {0}' for n in range(numsteps)]].astype('int64').values)
    return X_tensor, y_tensor

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

def Wang_Engel_Integral(T, T_min, T_opt, T_max):
    alpha = np.log(2)/np.log( (T_max - T_min)/(T_opt - T_min) )
    f_1 = (2*(np.sign(T - T_min)*(T - T_min))**(alpha + 1))*((T_opt - T_min)**alpha) / (alpha + 1)
    f_2 = ((np.sign(T - T_min)*(T - T_min))**((2*alpha) + 1)) / ((2*alpha) + 1)
    f_T = ( f_1 - f_2 ) / ((T_opt - T_min)**(2*alpha))
    f_T = np.nan_to_num(f_T)

    f_1_max = (2*(T_max - T_min)**(alpha + 1))*((T_opt - T_min)**alpha) / (alpha + 1)
    f_2_max = ((T_max - T_min)**((2*alpha) + 1)) / ((2*alpha) + 1)
    f_T_max = ( f_1_max - f_2_max ) / ((T_opt - T_min)**(2*alpha))
    return f_T*(T >= T_min)*(T<= T_max) + f_T_max*(T > T_max)
    
def Convolved_Wang_Engel(T, T_min, T_opt, T_max, gap = 4):
    return (1/(2*gap))*(Wang_Engel_Integral(np.minimum(T + gap, T_max), T_min, T_opt, T_max) - Wang_Engel_Integral(np.maximum(T - gap, T_min), T_min, T_opt, T_max))#
    
def integrand(T, T_min, T_opt, T_max, d, s, gap):
    #return modelling_fctns.Wang_Engel_Temp_response(T, T_min, T_opt, T_max, beta = 1.5)*np.exp(-((T - d)**2)/(2*(s**2)))
    return Convolved_Wang_Engel(T, T_min, T_opt, T_max, gap = gap)*(1/np.sqrt(2*np.pi*(s**2)))*np.exp(-((T - d)**2)/(2*(s**2)))
    
def expint(T_min, T_opt, T_max, d, s, gap):
    return quad(integrand, T_min, T_max, args=(T_min, T_opt, T_max, d, s, gap))[0]

vec_expint = np.vectorize(expint)

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

def fit_for_kf(epochs, model, loss_func, opt, train_dl, valid_dl, save_name = 'best_model', plot_opt = False, CNN=False, bce=False, verbose = False):
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
        if verbose and np.round((epoch/epochs)*100) % 10 == 0:
            print(epoch, train_loss, val_loss)
        #Save the model at the last epoch
        if epoch == epochs - 1:
            model_dir = 'C:\\Users\\wlwc1989\\Documents\\Phenology_Test_Notebooks\\ML_algorithms\\saved_models\\'
            model_path = os.path.join(model_dir, save_name + ".pt")
            torch.save({'epoch': best_epoch, 'model_state_dict': best_model_state}, model_path)
    #print(f'Loss: {best_loss}')
    if plot_opt:
        plot_train_val_loss(epochs, train_losses, val_losses, best_epoch)
    return model_loss

def K_fold_year_crossval(year_folds, X_tensor, y_tensor, 
                         model_class, epochs, bs, model_args, 
                         savename_prefix = 'KFold', 
                         freeze_params = False, loss = 'NLL', 
                         CNN=False, bce=False, lr = 0.01, 
                         MMD=False, n_channels = 4, 
                         DTF = False, obs_method = False, 
                         custom_loss = None,
                         GDD_init = None, verbose = False):
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
    for fold, year_fold in enumerate(year_folds):
        print(f"Fold {fold + 1}")

        # Define the data loaders for the current fold
        train_dl, val_dl = train_test_dl_by_year(X_tensor, y_tensor, test_years = year_fold, year_column = -2, bs = bs, n_channels = 8, train_proportion=0.8)

        # Initialize the model and optimizer
        model = model_class(*model_args).to(device)
        if GDD_init is not None:
            GDD_params = GDD_init[fold]#[0.31621224, 7.37066436, 22.88481424, 39.40562477]
            print(GDD_params)
            model = initialise_as_GDD(model, *GDD_params, n_channels=5)
        #GDD_params = [0.31621224, 7.37066436, 22.88481424, 39.40562477]
        #model = initialise_as_GDD(model, *GDD_params, n_channels=n_channels)
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

        model_loss = fit_for_kf(epochs, model, criterion, optimizer, 
                                train_dl, val_dl, save_name = savename_prefix + str(fold + 1), 
                                CNN=CNN, bce=bce, verbose=verbose)
        loss_values.append(model_loss)

        #Now look at stats for model:
        model_dir = 'C:\\Users\\wlwc1989\\Documents\\Phenology_Test_Notebooks\\ML_algorithms\\saved_models\\'
        model_path = os.path.join(model_dir, savename_prefix + str(fold + 1) + ".pt")
        checkpoint = torch.load(model_path, weights_only=True)
        TS_model = model_class(*model_args).to(device)
        TS_model.load_state_dict(checkpoint['model_state_dict'])

        comparison_frame = get_comparison_frame(savename_prefix + str(fold + 1), TS_model, val_dl, bce=bce, CNN=CNN, MMD=MMD, n_channels = n_channels, DTF = DTF, obs_method = obs_method)
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
        print(rsquared)
        bias = comparison_frame['observed'].mean() - comparison_frame['fitted'].mean()
        variance_modelled = comparison_frame[f'fitted'].var()
        
        R2s.append(rsquared)
        Biases.append(bias)
        Variances.append(variance_modelled)

    return loss_values, R2s, Biases, Variances, RMSEs, STDs, Corrs, Mins, LQs, Medians, UQs, Maxs

def K_fold_crossval(k_folds, train_ds, model_class, epochs, bs, model_args, savename_prefix = 'KFold', freeze_params = False, loss = 'NLL', CNN=False, bce=False, lr = 0.01, MMD=False, n_channels = 4, DTF = False, obs_method = False, custom_loss = None):
    ## LSTM_args: input_dim, hidden_dim, num_layers, output_dim
    # Initialize the k-fold cross validation
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
    for fold, (train_idx, test_idx) in enumerate(kf.split(train_ds)):
        print(f"Fold {fold + 1}")

        # Define the data loaders for the current fold
        train_dl = DataLoader(
            dataset=train_ds,
            batch_size=bs,
            sampler=torch.utils.data.SubsetRandomSampler(train_idx),
        )
        val_dl = DataLoader(
            dataset=train_ds,
            batch_size=bs,
            sampler=torch.utils.data.SubsetRandomSampler(test_idx),
        )

        # Initialize the model and optimizer
        model = model_class(*model_args).to(device)
        
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

        model_loss = fit_for_kf(epochs, model, criterion, optimizer, train_dl, val_dl, save_name = savename_prefix + str(fold + 1), CNN=CNN, bce=bce)
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
        
        R2s.append(rsquared)
        Biases.append(bias)
        Variances.append(variance_modelled)

    return loss_values, R2s, Biases, Variances, RMSEs, STDs, Corrs, Mins, LQs, Medians, UQs, Maxs

def K_fold_transfer(k_folds, ds, model_class, 
                    savename, epochs, bs, model_args, 
                    lr = 0.01, savename_prefix = 'KFold',
                    loss = 'NLL', 
                    CNN=False, bce=False, custom_loss = None, 
                    MMD = False, DTF = False, obs_method = False,
                    transfer_method = 'all wang'):
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
    statyear = ds[['Stations_id', 'Referenzjahr']].drop_duplicates()
    for fold, (train_idx, test_idx) in enumerate(kf.split(statyear)):
        train_statyear = statyear.iloc[train_idx]
        test_statyear = statyear.iloc[test_idx]
        train = ds.merge(train_statyear, on=['Stations_id', 'Referenzjahr'], how = 'inner') #.loc[ds[['Stations_id', 'year']].isin(train_statyear)]
        test = ds.merge(test_statyear, on=['Stations_id', 'Referenzjahr'], how = 'inner')
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
        model = model_class(*model_args).to(device)
        model_dir = 'C:\\Users\\wlwc1989\\Documents\\Phenology_Test_Notebooks\\ML_algorithms\\saved_models\\'
        model_path = os.path.join(model_dir, savename + ".pt")
        checkpoint = torch.load(model_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        for param in model.parameters():
            param.requires_grad = False
        if transfer_method == 'all wang':
            for param in [model.u0, model.u1, model.u2, model.u3]:
                param.requires_grad = True
        elif transfer_method == 'just rate':
            for param in [model.u4]:
                param.requires_grad = True
        elif transfer_method == 'wang and first':
            for param in [param for param in model.input_layer.parameters()] + [model.u0, model.u1, model.u2, model.u3]:
                param.requires_grad = True
        
        optimizer = optim.Adam(model.parameters(), lr=lr)
        if loss == 'NLL':
            criterion = nn.NLLLoss()
        elif loss == 'BCE':
            criterion = nn.BCELoss()
        elif loss == 'MSE':
            criterion = custom_loss
        # Train the model on the current fold

        #for param in [model.u0, model.u1, model.u2, model.u3, model.p0]:
        #    print(param.requires_grad, param.shape)
        model_loss = fit_for_kf(epochs, model, criterion, optimizer, train_dl, val_dl, save_name = savename_prefix + str(fold + 1), CNN=CNN, bce=bce, plot_opt=False, verbose=False)
        loss_values.append(model_loss)

        #Now look at stats for model:
        model_dir = 'C:\\Users\\wlwc1989\\Documents\\Phenology_Test_Notebooks\\ML_algorithms\\saved_models\\'
        model_path = os.path.join(model_dir, savename_prefix + str(fold + 1) + ".pt")
        checkpoint = torch.load(model_path, weights_only=True)
        TS_model = model_class(*model_args).to(device)
        TS_model.load_state_dict(checkpoint['model_state_dict'])
        # Now convert to days and get R^2 to compare to other models
        list_logs = []
        list_GDD_logs = []
        list_observed = []
        for xb, yb in val_dl:
            with torch.no_grad():
                if CNN:
                    if MMD:
                        list_logs.append(TS_model(xb)[0])
                        list_GDD_logs.append(TS_model(xb, no_nn=True)[0])
                    else:
                        list_logs.append(TS_model(xb))
                        list_GDD_logs.append(TS_model(xb, no_nn=True))
                else:
                    list_logs.append(TS_model(xb.transpose(1, 2)))
                    list_GDD_logs.append(TS_model(xb.transpose(1, 2), no_nn=True))
                #print(TS_model(xb.transpose(1, 2)))
            list_observed.append(yb)
        logs = torch.squeeze(torch.cat(list_logs))
        GDD_logs = torch.squeeze(torch.cat(list_GDD_logs))
        observed = torch.cat(list_observed)
        #print(logs.shape, torch.cat(list_logs, dim=0).shape)
        if not(DTF):
            if bce:
                fitted = torch.round(logs)
                fitted_GDD = torch.round(GDD_logs)
            else:
                fitted = torch.argmax(logs, dim=2)
                fitted_GDD = torch.argmax(GDD_logs, dim=2)
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
            GDD_days = L - fitted_GDD.sum(dim=1)
        
        comparison_frame = pd.DataFrame({'fitted': fitted_days.numpy().squeeze(), 'observed': observed_days.numpy().squeeze(), 'GDD fitted': GDD_days.numpy().squeeze()})
        comparison_frame['Error'] = comparison_frame['observed'] - comparison_frame['fitted']
        T_min = (9 + 10*(torch.tanh(TS_model.u1)) - 4).detach().cpu().numpy() #normally multiplier = 2
        T_opt = (28 + 7*(torch.tanh(TS_model.u2)) - 4).detach().cpu().numpy() #normally multiplier = 3
        T_max = (39 + 3.5*torch.tanh(TS_model.u3)).detach().cpu().numpy()#normally no multiplier
        scale = TS_model.u4.detach().cpu().numpy()
        threshold = TS_model.u0.detach().cpu().numpy()[0]
        x=[scale[0], T_min[0], T_opt[0], T_max[0]]
        #print(x, threshold)
        #print(test.columns.to_list())
        #cf2 = run_GDD(x, test.rename(columns={'Referenzjahr': 'year'}), 't2m', response_type='Wang', thresholds =[threshold], growing_period_length = 120)
        #print(cf2['modelled time to beginning of flowering'])
        #comparison_frame['GDD_fitted'] = cf2['modelled time to beginning of flowering'].values
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

    return comparison_frame_full, loss_values, R2s, Biases, Variances, RMSEs, SSEs, STDs, Corrs, Mins, LQs, Medians, UQs, Maxs, lengths, obsvars

def K_fold_transfer_regions(ds_dict, transfer_method = 'just rate'):
    bs = 30000
    n_channels = 5
    input_dim = n_channels # Example: 100-dimensional word embeddings
    hidden_dim = 16#day2 -> 8#day -> 16 #day3 ->4
    num_layers = 4#day3 -> 8
    output_dim = 1  # Example: 5 classes
    KG = False
    KG2 = True
    CNN = False
    bce=True

    model_args = (input_dim, hidden_dim, num_layers, output_dim, KG, KG2)
    save_name = 'NN_response_DE_no_day5'
    lr = 0.01
    epochs = 500
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
        comparison_frame, loss_values, R2s, Biases, Variances, RMSEs, SSEs, STDs, Corrs, Mins, LQs, Medians, UQs, Maxs, lengths, obsvars = K_fold_transfer(5, ds, nn_temp_response, 
                                                                                                                save_name, epochs, bs, model_args,lr = lr, 
                                                                                                                loss = 'MSE', CNN=CNN, bce=bce, 
                                                                                                                custom_loss=criterion, DTF = False, transfer_method=transfer_method)
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

def get_comparison_frame(savename, TS_model, dl, bce=False, CNN=False, MMD=False, n_channels = 4, DTF = False, obs_method = False):
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
            list_years.append(xb[:, n_channels, 0])
            list_stations.append(xb[:, n_channels + 1, 0])
            #print(TS_model(xb.transpose(1, 2)))
        list_observed.append(yb)
    logs = torch.squeeze(torch.cat(list_logs))
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
        observed_days = observed[:, 0]
        observed_days = 90 + 100 - observed[:, 90]
        if obs_method:
            fitted_days = observed_days + logs[range(0, logs.shape[0]), observed_days.int()]
        else:
            fitted_days = mindex_at_0(logs)
            fitted_days = 90 + 100 - logs[:, 90]
            #fitted_days = logs[:, 90]
        print(torch.sum(fitted_days < 30), ' fits too small')
    else:
        #print(len(fitted.shape))
        if len(fitted.shape) == 1:
            fitted_days = len(fitted) - fitted.sum()
            observed_days = len(observed) - observed.sum()
        else:
            L = fitted.shape[1]
            fitted_days = L - fitted.sum(dim=1)
            observed_days = L - observed.sum(dim=1)
    if len(fitted.shape) > 1:
        comparison_frame = pd.DataFrame({'fitted': fitted_days.numpy().squeeze(), 'observed': observed_days.numpy().squeeze()})
    else:
        comparison_frame = pd.DataFrame({'fitted': [fitted_days.numpy().squeeze()], 'observed': [observed_days.numpy().squeeze()]})
    #print(stations.numpy().squeeze())
    return comparison_frame
    
#def get_comparison_frame(savename, TS_model, dl, bce=False, CNN=False):
#    model_dir = 'C:\\Users\\wlwc1989\\Documents\\Phenology_Test_Notebooks\\ML_algorithms\\saved_models\\'
#    model_path = os.path.join(model_dir, savename + ".pt")
#    checkpoint = torch.load(model_path, weights_only=True)
#    TS_model.load_state_dict(checkpoint['model_state_dict'])
#    list_logs = []
#    list_observed = []
#    list_stations = []
#    list_years = []
#    for xb, yb in dl:
#        with torch.no_grad():
#            if CNN:
#                list_logs.append(TS_model(xb))
#            else:
#                list_logs.append(TS_model(xb.transpose(1, 2)))
#            list_years.append(xb[:, 4, 0])
#            list_stations.append(xb[:, 5, 0])
#            #print(TS_model(xb.transpose(1, 2)))
#        list_observed.append(yb)
#    logs = torch.squeeze(torch.cat(list_logs))
#    stations = torch.squeeze(torch.cat(list_stations))
#    years = torch.squeeze(torch.cat(list_years))
#    if bce:
#        fitted = torch.round(logs)
#    else:
#        fitted = torch.argmax(logs, dim=2)
#    L = fitted.shape[1]
#    fitted_days = L - fitted.sum(dim=1)
#    observed = torch.cat(list_observed)
#    observed_days = L - observed.sum(dim=1)
#    #print(stations.numpy().squeeze())
#    comparison_frame = pd.DataFrame({'Stations_id': stations.numpy().squeeze(), 'year': years.numpy().squeeze(), 'fitted': fitted_days.numpy().squeeze(), 'observed': observed_days.numpy().squeeze()})
#    return comparison_frame
    
def subset_ds_from_tensors(x_train, y_train, train_proportion = 0.2):
    full_ds = TensorDataset(x_train, y_train)
    train_size = int(train_proportion * len(full_ds))
    val_size = int(len(full_ds) - (1/train_proportion - 1)*train_size)
    split_ds = random_split(full_ds, [train_size, train_size, train_size, train_size, val_size])
    return split_ds

def subset_dl_from_tensors(x_train, y_train, bs = 500, n_channels=1, train_proportion = 0.2):
    split_dl = []
    split_ds = subset_ds_from_tensors(x_train, y_train, train_proportion = train_proportion)
    for ds in split_ds:
        dl = DataLoader(ds, batch_size=bs)
        dl = WrappedDataLoader(dl, lambda x, y: preprocess(x, y, channels =n_channels))
        split_dl.append(dl)
    return split_dl

def loss_batch(model, loss_func, xb, yb, opt=None, CNN=False, bce=False):
    if CNN:
        outputs = model(xb)
        loss = loss_func(torch.squeeze(outputs.transpose(1, 2)), torch.squeeze(yb.float()))
    elif bce:
        outputs = model(xb.transpose(1, 2))
        loss = loss_func(torch.squeeze(outputs.float()), torch.squeeze(yb.float()))#.transpose(1, 2)
    else:
        outputs = model(xb.transpose(1, 2))
        loss = loss_func(outputs.transpose(1, 2), yb.long())

    if opt is not None:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)
def fit_on_subsets(x_train, y_train, model_class, epochs, bs, model_args, savename_prefix = 'subset_coeffs', freeze_params = False, loss = 'NLL', CNN=False, bce=False):
    split_dl = subset_dl_from_tensors(x_train, y_train)
    coeffs_list = []
    loss_values = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for fold, train_dl in enumerate(split_dl):
        
        val_dl = split_dl[fold - 1]
        print(f"Fold {fold + 1}")
        # Initialize the model and optimizer
        model = model_class(*model_args).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        if loss == 'NLL':
            criterion = nn.NLLLoss()
        elif loss == 'BCE':
            criterion = nn.BCELoss()
        # Train the model on the current fold
        if freeze_params:
            for param in [model.u0, model.u1, model.u2, model.u3, model.p0]:
                param.requires_grad = False
        for xb, yb in train_dl:
            batch_loss, batch_len = loss_batch(model, criterion, xb, yb, optimizer, CNN=CNN, bce=bce)
        model_loss = fit_for_kf(epochs, model, criterion, optimizer, train_dl, val_dl, save_name = savename_prefix + str(fold + 1), CNN=CNN, bce=bce)
        loss_values.append(model_loss)

        #Now look at stats for model:
        model_dir = 'C:\\Users\\wlwc1989\\Documents\\Phenology_Test_Notebooks\\ML_algorithms\\saved_models\\'
        model_path = os.path.join(model_dir, savename_prefix + str(fold + 1) + ".pt")
        checkpoint = torch.load(model_path, weights_only=True)
        TS_model = model_class(*model_args).to(device)
        TS_model.load_state_dict(checkpoint['model_state_dict'])

        coeffs = []
        for name, param in TS_model.named_parameters():
            if param.requires_grad:
                print(name, param.data)
                coeffs.append(param.data)
        coeffs = coeffs[:-4]

        def B_0(u):
            return ((1 - u**2)**2)*(u >= -1)*(u <= 1) #np.maximum((1/6)*(-(x**3) + 3*(x**2) - 3*x + 1), 0)

        def response(meantemp):
            resp = 0
            for i, coeff in enumerate(coeffs):
                resp += torch.abs(coeff)*B_0(0.25*(meantemp - i*2))
            return resp
            
        fig, ax = plt.subplots()
        ax.plot(np.arange(0, 45, 0.5), response(np.arange(0, 45, 0.5)))
        maxval = np.arange(0, 45, 0.5)[response(np.arange(0, 45, 0.5)).argmax()]
        print(maxval)
        ax.axvline(maxval)
        ax.axhline(0)
        coeffs_list.append(coeffs)
    return coeffs_list
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
    
class nn_temp_response(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, KG=False, KG2 = False):
        super(nn_temp_response, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.KG = KG
        self.KG2 = KG2

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False) # batch_first handles input shape (batch, seq, features)
        if self.KG2:
            self.input_layer = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4= nn.Linear(hidden_dim, hidden_dim)
        self.layer5 = nn.Linear(hidden_dim, hidden_dim)
        self.layer6= nn.Linear(hidden_dim, hidden_dim)
        self.layer7 = nn.Linear(hidden_dim, hidden_dim)
        self.layer8 = nn.Linear(hidden_dim, hidden_dim)
        self.layer9 = nn.Linear(hidden_dim, hidden_dim)
        self.layer10= nn.Linear(hidden_dim, hidden_dim)
        self.layer11 = nn.Linear(hidden_dim, hidden_dim)
        self.layer12 = nn.Linear(hidden_dim, hidden_dim)
        self.layer13= nn.Linear(hidden_dim, hidden_dim)
        self.layer14 = nn.Linear(hidden_dim, hidden_dim)
        self.layer15 = nn.Linear(hidden_dim, hidden_dim)
        self.layer16 = nn.Linear(hidden_dim, hidden_dim)
        self.layer17 = nn.Linear(hidden_dim, hidden_dim)
        self.layer18 = nn.Linear(hidden_dim, hidden_dim)
        self.layers_list = [self.input_layer, self.layer1, self.layer2, self.layer3,
                            self.layer4, self.layer5, self.layer6,
                            self.layer7, self.layer8, self.layer9,
                            self.layer10, self.layer11, self.layer12,
                            self.layer13, self.layer14, self.layer15,
                            self.layer16, self.layer17, self.layer18]
        self.fc = nn.Linear(hidden_dim, output_dim)  # Fully connected layer for classification
        self.input_dim = input_dim
        self.sig = nn.Sigmoid()

        #self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.activation = nn.Tanh()

        self.u0 = torch.nn.Parameter(torch.Tensor([20])) #Now TTR
        self.u1 = torch.nn.Parameter(torch.Tensor([0])) #not used
        self.u2 = torch.nn.Parameter(torch.Tensor([0])) # not used
        self.u3 = torch.nn.Parameter(torch.Tensor([0])) #not used
        self.u4 = torch.nn.Parameter(torch.Tensor([50]))

        self.p0 = torch.nn.Parameter(torch.Tensor([13]))
        

    def forward(self, x0, no_nn = False):
        x = x0[:, :, :self.input_dim]
        original_temp = x0[:, :, [0]]
        x2 = x0[:, :, [0]]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device) # Initialize hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device) # Initialize cell state
        # Apply Wang Engel
        #T_min = 9 + 8*(torch.tanh(self.u1)) - 4 #was 2*
        #T_opt = 28 + 8*(torch.tanh(self.u2)) - 4 #was 3*
        #T_max = 39 + 2*torch.tanh(self.u3) - 2 #was 1*
        T_min = 9 + 10*(torch.tanh(self.u1)) - 4 #normally multiplier = 2
        T_opt = 28 + 7*(torch.tanh(self.u2)) - 4 #normally multiplier = 3
        T_max = 39 + 3.5*torch.tanh(self.u3)#normally no multiplier
        alpha = np.log(2)/torch.log( (T_max - T_min)/(T_opt - T_min) )
        beta = 1
        #print(x.shape)
        #print(T_min, T_opt, T_max)
        #print(alpha)
        #print(((2*(x - T_min)*(x >= T_min))**alpha))
        if self.KG:
            x[:, :, 0] = (x[:, :, 0] <= T_max) * ( (2*(((x[:, :, 0] - T_min)*(x[:, :, 0] >= T_min) + (x[:, :, 0] <= T_min).float()).pow(alpha))*(x[:, :, 0] >= T_min))*((T_opt - T_min).pow(alpha)) - ((((x[:, :, 0] - T_min)*(x[:, :, 0] >= T_min) + (x[:, :, 0] <= T_min).float()).pow(2*alpha))*(x[:, :, 0] >= T_min)) ) / ((T_opt - T_min).pow(2*alpha))
        elif self.KG2:
            x2[:, :, 0] = (x2[:, :, 0] <= T_max) * ( (2*(((x2[:, :, 0] - T_min)*(x2[:, :, 0] >= T_min) + (x2[:, :, 0] <= T_min).float()).pow(alpha))*(x2[:, :, 0] >= T_min))*((T_opt - T_min).pow(alpha)) - ((((x2[:, :, 0] - T_min)*(x2[:, :, 0] >= T_min) + (x2[:, :, 0] <= T_min).float()).pow(2*alpha))*(x2[:, :, 0] >= T_min)) ) / ((T_opt - T_min).pow(2*alpha))
        #x = torch.nan_to_num(x)
        #x = x*(x >= T_min)*(x<= T_max)
        #print(x)
        #print(x, x.shape)
        # Make cumulative
        if self.input_dim >= 3 and self.KG:
            x[:, :, 1] = 0.5*(1 + torch.tanh(2*(x[:, :, 1] - self.p0)))
        # Forward pass through LSTM
        #x = torch.swapaxes(x, 1, 2)
        #for i, linlayer in enumerate([self.input_layer, self.layer1, self.layer2, self.layer3, 
        #                              self.layer4, self.layer5, self.layer6,
        #                              self.layer7, self.layer8, self.layer9,
        #                              self.layer10, self.layer11, self.layer12]):
        for i, linlayer in enumerate(self.layers_list[:self.num_layers]):
            #print(i)
            #print(x.shape)
            #print(conv)
            x = linlayer(x)
            x = self.activation(x)
            #x = x[:, :, :-conv.padding[0]]
        #torch.nn.init.xavier_uniform(self.fc.weight)
        # Get the hidden state of the last time step
        # output[:, -1, :] is more efficient for batch_first=True
        #last_hidden = output[:, -1, :] # (batch, hidden_dim)
        #print(self.fc(output))
        
        # Classify all layers using fully connected layer
        if self.KG2:
            #print(torch.any(self.fc(x) != 1))
            if no_nn:
                out_space = torch.abs(x2)*(original_temp >5)# make sure it is non-neg and 0 for low temps (batch, output_dim)
            else:
                out_space = torch.abs(x2 + self.fc(x))*(original_temp >5)# make sure it is non-neg and 0 for low temps (batch, output_dim)
            #print((out_space - x2).squeeze()[0])
        else:
            out_space = torch.abs(self.fc(x))*(original_temp >5)# make sure it is non-neg and 0 for low temps (batch, output_dim)
        out_space = self.u4*torch.cumsum(out_space, dim=1) #x[:, :, 0] = self.u0*torch.cumsum(x[:, :, 0], dim = 1)
        if self.output_dim == 1:
            out_scores = self.sig(out_space - self.u0)
        else:
            out_scores = F.log_softmax(out_space, dim=2)
        #print(out_scores[0, :, :])
        #print(out_scores[0, :])
        return out_scores
    
def load_model(savename, model):
    model_dir = 'C:\\Users\\wlwc1989\\Documents\\Phenology_Test_Notebooks\\ML_algorithms\\saved_models\\'
    model_path = os.path.join(model_dir, savename + ".pt")
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    #print(checkpoint['model_state_dict'])
    return model
def cardinal_temps_to_ML_params(T_min, T_opt, T_max):
    u1 = np.arctanh((T_min - 5)/10)
    u2 = np.arctanh((T_opt - 24)/7)
    u3 = np.arctanh((T_max - 39)/3.5)
    return u1, u2, u3

def initialise_as_GDD(model, scale, T_min, T_opt, T_max, n_channels = 1):
    #for linlayer in model.layers_list:
    #    nn.init.eye_(linlayer.weight)
    #    nn.init.zeros_(linlayer.bias)
    #nn.init.zeros_(model.fc.weight)
    #model.fc.weight = torch.nn.Parameter(model.fc.weight.data)
    nn.init.zeros_(model.fc.bias)
    #nn.init.zeros_(model.input_layer.weight)
    model.u4 = nn.Parameter(torch.tensor([scale]).float())
    u1, u2, u3 = cardinal_temps_to_ML_params(T_min, T_opt, T_max)
    model.u1 = nn.Parameter(torch.tensor([u1]))#.float())
    model.u2 = nn.Parameter(torch.tensor([u2]))#.float())
    model.u3 = nn.Parameter(torch.tensor([u3]))#.float())
    return model

def initialise_as_GDD2(model, scale, T_min, T_opt, T_max, n_channels = 1):
    for linlayer in model.layers_list:
        nn.init.eye_(linlayer.weight)
        nn.init.zeros_(linlayer.bias)
    nn.init.zeros_(model.fc.weight)
    #model.fc.weight = torch.nn.Parameter(model.fc.weight.data/model.hidden_dim)
    nn.init.ones_(model.fc.bias)
    #nn.init.zeros_(model.input_layer.weight)
    model.u4 = nn.Parameter(torch.tensor([scale]).float())
    u1, u2, u3 = cardinal_temps_to_ML_params(T_min, T_opt, T_max)
    model.u1 = nn.Parameter(torch.tensor([u1]))#.float())
    model.u2 = nn.Parameter(torch.tensor([u2]))#.float())
    model.u3 = nn.Parameter(torch.tensor([u3]))#.float())
    return model

def split_ds_by_AEZ2(ds):
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