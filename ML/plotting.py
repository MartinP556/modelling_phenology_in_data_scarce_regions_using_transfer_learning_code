import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
from itertools import compress
import dataset_fctns

def count_transition_numbers(phen_data):
    for phase_name in phen_data['Name of phase'].unique():
        for next_phase in phen_data['Next stage name'].unique():
            obs_this_change = phen_data.where((phen_data['Name of phase'] == phase_name)&(phen_data['Next stage name'] == next_phase)).dropna()
            if len(obs_this_change) != 0:
                print('Number going from ', phase_name, ' to ', next_phase, ': ', len(obs_this_change))
                print('Average time going from ', phase_name, ' to ', next_phase, ': ', obs_this_change['Time to next stage'].mean())

def Germany_plot(figsize = (10,10)):
    adm1_shapes = list(shpreader.Reader('gadm41_DEU_1/gadm41_DEU_1.shp').geometries())
    fig,ax=plt.subplots(figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})
    ax.coastlines(resolution='10m')
    ax.add_geometries(adm1_shapes, ccrs.PlateCarree(), edgecolor='black', facecolor='none', alpha=1)
    ax.set_extent([5, 16, 47, 56], ccrs.PlateCarree())
    return fig, ax

def scatter_plot_Germany(lats, lons,  plot_colors = 'r', title = 'Stations', font_size = 20, save_name='mistake_plot', colorbar = True):
    adm1_shapes = list(shpreader.Reader('gadm41_DEU_1/gadm41_DEU_1.shp').geometries())
    fig,ax=Germany_plot()
    if colorbar:
        plot_Germany = ax.scatter(lons, lats, s = 20, c = plot_colors, cmap = 'Purples')
        cbar = fig.colorbar(plot_Germany, shrink = 0.5)
        cbar.ax.tick_params(labelsize=15)
        cbar.set_label('Average day of emergence\n(from 1st Jan)', size = font_size)
    else:
        plot_Germany = ax.scatter(lons, lats, s = 20, c = plot_colors, cmap = 'Purples')
    fig.savefig('plots/' + save_name + '.png')

def heatmap_Germany(lats, lons, Zs, title = 'Stations', font_size = 20, save_name='mistake_plot', colorbar = True):
    adm1_shapes = list(shpreader.Reader('gadm41_DEU_1/gadm41_DEU_1.shp').geometries())
    fig,ax=plt.subplots(figsize=(10,10), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.coastlines(resolution='10m')
    ax.add_geometries(adm1_shapes, ccrs.PlateCarree(), edgecolor='black', facecolor='none', alpha=1)
    ax.set_extent([5, 16, 47, 56], ccrs.PlateCarree())
    ax.set_title(title, fontsize = font_size)

def plot_station_locations(phen_data, station_data, font_size = 20):
    phen_data = dataset_fctns.add_locations(phen_data, station_data)
    latlons = phen_data.drop_duplicates(['lat', 'lon'])
    station_lats = latlons['lat']
    station_lons = latlons['lon']
    scatter_plot_Germany(station_lats, station_lons, title = 'Station locations', save_name='Station_locations', colorbar = False, font_size=font_size)

def day_of_emergence_map(phen_data, binsize = 0.25, font_size = 20):
    lons = []
    lats = []
    emergence_days = []
    for LON in np.arange(5, 16, binsize):
        for LAT in np.arange(47, 56, binsize):
            #mask = (phen_data['Name of phase'] == 'beginning of emergence')*np.greater_equal(phen_data['lat'], LAT)*np.less(phen_data['lat'], LAT + binsize)*np.greater_equal(phen_data['lon'], LON)*np.less(phen_data['lon'], LON + binsize)
            mask = (phen_data['Name of phase'] == 'beginning of emergence')*(phen_data['lat'] >= LAT)*(phen_data['lat'] < LAT + binsize)*(phen_data['lon'] >= LON)*(phen_data['lon'] < LON + binsize)
            if len(phen_data[mask]) != 0:
                average_emergence_day = phen_data[mask]['Jultag'].dropna().mean()
                lons.append(LON)
                lats.append(LAT)
                emergence_days.append(average_emergence_day)
    scatter_plot_Germany(lats, lons, plot_colors = emergence_days, title = 'Average day of emergence\n(from 1st Jan)', font_size = font_size, save_name='day_of_emergence_map')
    
def plot_num_obs_grouped(phen_data, grouping_variable, save_name, title, font_size = 20, xlab='Number of observations', ylab = 'Number of stations'):
    fig, ax = plt.subplots(figsize = (10, 10))
    ax.hist(phen_data.groupby(grouping_variable).size(), bins = 30)
    ax.set_title(title, fontsize = font_size)
    ax.set_xlabel(xlab, fontsize = font_size)
    ax.set_ylabel(ylab, fontsize = font_size)
    fig.savefig('plots/' + save_name + '.png')

def plot_stage_to_stage(phen_data, stage1, stage2, winter_sowing = False, font_size = 20):
    ##Plan: multi-index with year and station.
    ## Isolate twice with .where to get the times of the two stages
    ## Subtract and hopefully throw up a NaN where one or both are missing
    intervals = dataset_fctns.time_stage_to_stage(phen_data, stage1, stage2, winter_sowing=winter_sowing)
    fig, ax = plt.subplots(figsize = (10, 5))
    ax.hist(intervals.dropna(), bins = 250)
    ax.set_title(f'Length of time from {stage1}\nto {stage2}', fontsize = font_size)
    ax.set_ylabel('Number of observations', fontsize = font_size)
    ax.set_xlabel(f'Length of time from {stage1}\nto {stage2}', fontsize = font_size)
    ax.set_xlim(0, 365)
    ax.annotate(f'Mean: {np.round(intervals.mean(), decimals=2)}\nStd: {np.round(intervals.std(), decimals=2)}', xy=(.8, .8), xycoords='axes fraction', fontsize = font_size - 3)#, fontsize=14, transform=plt.gcf().transFigure)
    fig.savefig(f'plots/{stage1}_to_{stage2}.png')


def plot_avg_num_yearly_obs(phen_data, font_size = 20):
    fig, ax = plt.subplots(figsize = (10, 10))
    obsnums_yearly = phen_data.groupby(['Stations_id', 'Referenzjahr']).size()
    ax.hist(obsnums_yearly.groupby('Stations_id').mean(), bins = 30)
    ax.set_title('Average number of observations per year', fontsize = font_size)
    ax.set_xlabel('Number of stations', fontsize = font_size)
    ax.set_ylabel('Average number of obs per year', fontsize = font_size)
    fig.savefig('plots/avg_num_yearly_obs.png')

def length_phase_box_plot(phen_data, phase_names, font_size = 20):
    fig, ax = plt.subplots(figsize = (10, 10))
    phase_lengths = []
    for phase_name in phase_names:
        obs_this_phase = phen_data[phen_data['Name of phase'] == phase_name]
        phase_lengths.append(obs_this_phase['Time to next stage'].dropna().dt.days.values)
    ax.boxplot(phase_lengths, tick_labels=phase_names, widths = 0.5) #positions = [obs_this_phase['Stations_id'].unique()[0]],
    ax.set_ylim(0, 100)
    plt.xticks(rotation = 90)
    ax.tick_params(labelsize = font_size)
    ax.set_title('Length of phases', fontsize = font_size)
    ax.set_ylabel('Length of phase (days)', fontsize = font_size)
    fig.savefig('plots/length_phase_box_plot.png', bbox_inches='tight')

def hist2d_locations(lats, lons, bin_num=20, font_size = 20):
    hist, lonedges, latedges = np.histogram2d(lons, lats, bins=bin_num)#, range=[[0, 4], [0, 4]])
    sizelat = (latedges[1] - latedges[0])/2
    sizelon = (lonedges[1] - lonedges[0])/2
    lonpos, latpos = np.meshgrid(lonedges, latedges, indexing="ij")#np.meshgrid(lonedges[:-1] + sizelon, latedges[:-1] + sizelat, indexing="ij")
    lonpos = lonpos#.ravel()
    latpos = latpos#.ravel()
    fig, ax = Germany_plot()
    ax.set_title('Frequency of observations of ripeness time by location', fontsize = font_size)
    densities = ax.pcolormesh(lonpos, latpos, hist, cmap='Purples')
    cbar = plt.colorbar(densities, fraction = 0.03)
    cbar.set_label(label = 'Number of reports in 2007', size=font_size)
    cbar.ax.tick_params(labelsize=font_size - 2) 
    plt.show()

def plot_obs_per_year(ds_obs, save_name, phase_list = [], font_size = 20):
    fig, ax = plt.subplots(figsize = (10, 5))
    yearly_counts = ds_obs.groupby('Referenzjahr').count()
    yearly_counts = ds_obs.groupby('Referenzjahr').count()
    yearly_counts.index = pd.to_datetime(yearly_counts.index, format='%Y')#.resample('str') #np.datetime64(yearly_counts.index, 'Y')
    yearly_counts = yearly_counts.resample('YS').asfreq()
    if len(phase_list) != 0:
        for phase_index, phase in enumerate(phase_list):
            ax.plot(yearly_counts.index, yearly_counts[yearly_counts.columns[phase_index]], color = f'C{phase_index}', label = phase)
    ax.set_xlabel('Year', fontsize = font_size)
    ax.set_ylabel('Number of observations', fontsize = font_size)
    fig.suptitle('Number of observations of phase by year', fontsize = font_size)
    ax.legend(fontsize = font_size - 2, bbox_to_anchor=(0.7, -0.2))
    fig.savefig(f'C:\\Users\\wlwc1989\\Documents\\Phenology_Test_Notebooks\\phenology_dwd\\plots\\{save_name}.png')

def box_plot_modelled_observed(ds, phases, font_size = 20):
    for phase in phases:
        fig, ax = plt.subplots(figsize = (10, 10))
        ax.boxplot([ds[f'modelled time emergence to {phase}'].dropna(), ds[f'ML prediction emergence to {phase}'].dropna(), ds[f'observed time emergence to {phase}'].dropna()], 
                   tick_labels=[f'modelled time emergence to\n{phase}', f'ML prediction emergence to\n{phase}', f'observed time emergence to\n{phase}'], 
                   widths = 0.5, showfliers=False) #positions = [obs_this_phase['Stations_id'].unique()[0]],
        #ax.set_ylim(0, 100)
        plt.xticks(rotation = 90)
        ax.tick_params(labelsize = font_size)
        ax.set_title(f'Modelled and observed times to {phase}', fontsize = font_size)
        ax.set_ylabel('Time (days)', fontsize = font_size)
        fig.savefig(f'plots/ML_modelled_observed_{phase}.png', bbox_inches='tight')

        #fig, ax = plt.subplots(figsize = (10, 10))
        #ax.boxplot((ds[f'modelled time emergence to {phase}']- ds[f'observed time emergence to {phase}']).dropna(), 
        #           tick_labels=[f'diff modelled/observed time emergence\nto{phase}'], 
        #           widths = 0.5, showfliers=False) #positions = [obs_this_phase['Stations_id'].unique()[0]],
        #ax.set_ylim(0, 100)
        #plt.xticks(rotation = 90)
        #ax.tick_params(labelsize = font_size)
        #ax.set_title(f'Difference between modelled\nand observed times to {phase}', fontsize = font_size)
        #ax.set_ylabel('Time (days)', fontsize = font_size)
        #fig.savefig(f'plots/modelled_observed_{phase}_diffs.png', bbox_inches='tight')


def plot_error_distn(ds, phases, training_means, font_size = 20):
    for phase_index, phase in enumerate(phases):
        fig, ax = plt.subplots(figsize = (10, 7))
        print(training_means[phase_index])
        residuals_to_average = training_means[phase_index] - ds[f'observed time emergence to {phase}']
        ML_residuals = ds[f'ML prediction emergence to {phase}'] - ds[f'observed time emergence to {phase}']
        model_residuals = ds[f'modelled time emergence to {phase}'] - ds[f'observed time emergence to {phase}']
        ax.boxplot([model_residuals.dropna(), ML_residuals.dropna(), residuals_to_average.dropna()],
                   tick_labels=[f'residuals modelled\ntime', f'residuals ML\nprediction', f'obs - training mean'], #tick_labels=[f'residuals modelled time emergence\nto {phase}', f'residuals ML prediction emergence\nto {phase}', f'residuals compared to training mean\n{phase}'],
                   widths = 0.5, showfliers=False) #positions = [obs_this_phase['Stations_id'].unique()[0]],
        #ax.set_ylim(0, 100)
        plt.xticks(rotation = 50)
        ax.tick_params(labelsize = font_size)
        ax.set_title(f'ML and model residuals,\ntime to {phase}', fontsize = font_size)
        ax.set_ylabel('Time (days)', fontsize = font_size)
        fig.savefig(f'plots/ML_modelled_observed_{phase}.png', bbox_inches='tight')