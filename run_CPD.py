#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 08:50:09 2018

@author: ian
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb

from QCCPD import fit

def barr_nomenclature(stats_df):
    
    print '\nxs2:\n'
    print '    n = {}'.format(str(int(stats_df.n)))
    print '    Cp = {}'.format(str(round(float(stats_df.ustar_th_b), 5)))
    print '    Fmax = {}'.format(str(round(float(stats_df.fmax_b), 5)))
    print '    p = {}'.format(str(round(float(stats_df.p_b), 5)))
    print '    b0 = {}'.format(str(round(float(stats_df.b0), 5)))
    print '    b1 = {}'.format(str(round(float(stats_df.b1), 5)))
    print '    b2 = {}'.format('NaN')
    print '    c2 = {}'.format('NaN')
    print '    cib0 = {}'.format('Not implemented')
    print '    cib1 = {}'.format('Not implemented')
    print '    cib2 = {}'.format('NaN')

    print '\nxs3:\n'
    print '    n = {}'.format(str(int(stats_df.n)))
    print '    Cp = {}'.format(str(round(float(stats_df.ustar_th_a), 5)))
    print '    Fmax = {}'.format(str(round(float(stats_df.fmax_a), 5)))
    print '    p = {}'.format(str(round(float(stats_df.p_a), 5)))
    print '    b0 = {}'.format(str(round(float(stats_df.a0), 5)))
    print '    b1 = {}'.format(str(round(float(stats_df.a1), 5)))
    print '    b2 = {}'.format('What is this?!')
    print '    c2 = {}'.format(str(round(float(stats_df.a2), 5)))
    print '    cib0 = {}'.format('Not implemented')
    print '    cib1 = {}'.format('Not implemented')
    print '    cib2 = {}'.format('Not implemented')

def get_data(path_to_file):
    
    df = pd.read_csv(path_to_file)
    df.drop('Unnamed: 0', axis = 1, inplace = True)
    return df

def plot_fits(temp_df, stats_df):
    
    if any(np.isnan(stats_df.cp_a) | np.isnan(stats_df.cp_b)):
        print '\nUnsuccessful fits... aborting plot routine'
        return
    
    # Create series for operational (b) model
    change_point_b_model = stats_df['cp_b'].iloc[0]
    threshold_b_model = temp_df['ustar'].iloc[change_point_b_model]
    ustar_b_series = temp_df['ustar'].copy()
    ustar_b_series.iloc[change_point_b_model + 1:] = threshold_b_model
    yhat_b_series = stats_df['b0'].iloc[0] + stats_df['b1'].iloc[0] * ustar_b_series 

    # Create series for diagnostic (a) model    
    change_point_a_model = stats_df['cp_a'].iloc[0]
    threshold_a_model = temp_df['ustar'].iloc[change_point_a_model]
    ustar_ax1_series = temp_df['ustar'].copy()
    ustar_ax1_series.iloc[change_point_a_model + 1:] = threshold_a_model
    dummy_series = np.concatenate([np.zeros(change_point_a_model), 
                                   np.ones(len(temp_df) - change_point_a_model)])
    ustar_ax2_series = (temp_df['ustar'] - threshold_a_model) * dummy_series
    yhat_a_series = (stats_df['a0'].iloc[0] + 
                     stats_df['a1'].iloc[0] * ustar_ax1_series + 
                     stats_df['a2'].iloc[0] * ustar_ax2_series)

    # Now plot    
    fig, ax = plt.subplots(1, figsize = (12, 8))
    fig.patch.set_facecolor('white')
    ax.set_xlim([0, round(max(temp_df.ustar) * 1.05, 2)])
    ax.set_ylim([0, round(max(temp_df.Fc) * 1.05, 2)])
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis = 'x', labelsize = 14)
    ax.tick_params(axis = 'y', labelsize = 14)
    ax.set_xlabel('$u*\/(m\/s^{-1}$)', fontsize = 16)
    ax.set_ylabel('$F_c\/(\mu mol C\/m^{-2} s^{-1}$)', fontsize = 16)
    ax.plot(temp_df['ustar'], temp_df['Fc_clean'], color = 'black', 
            label = 'Noise-free synthetic')
    ax.plot(temp_df['ustar'], temp_df['Fc'], 'bo', label = 'Noisy synthetic')
    ax.plot(temp_df['ustar'], yhat_b_series, color = 'magenta', 
            label = 'Operational model fit')   
    ax.plot(temp_df['ustar'], yhat_a_series, color = 'cyan', 
            label = 'Diagnostic model fit')   
    props = dict(boxstyle='round,pad=1', facecolor='white', alpha=0.5)
    txt = ('Change point detected at u* = ' + 
           str(round(stats_df['ustar_th_b'],3)) +
           ' (i = '+str(change_point_b_model) + ')')
    ax.text(0.57, 0.1, txt, bbox = props, fontsize = 12, 
            verticalalignment = 'top', transform = ax.transAxes)
    ax.legend(loc = [0.1, 0.8], frameon = False, fontsize = 12)


# Set parameters for generation of noisy data
noise_dict = {'slope': 0.44, 
              'intercept': 0.4}
intercept = 0
slope = 20
change_point = 15
n = 55
use_saved = True
noise_on = True
save_to_file = True
path = '/home/ian/Documents/Test/data.csv'

# Generate Laplacian noise using estimate of standard deviation of noise as 
# scaling parameter, then superimpose on 'clean' flux signal
if use_saved:

    df = get_data(path)
    
# Or - load a saved .csv file containing the results of previously saved 
# synthetic data generation
else:

    ustar_series = np.linspace(0, 0.6, n)
    flux_series = np.concatenate((ustar_series[:change_point] * slope + 
                                  intercept,
                                  np.tile(ustar_series[change_point] * slope + 
                                          intercept, 
                                          n - change_point)))
    sigma_delta_series = (flux_series * noise_dict['slope'] + 
                          noise_dict['intercept'])
    error_series = np.random.laplace(0, sigma_delta_series / np.sqrt(2))
    df = pd.DataFrame({'ustar': ustar_series, 'Fc_clean': flux_series})
    df['Fc'] = flux_series + error_series if noise_on else flux_series

    if save_to_file: df.to_csv(path)
    
stats_df = pd.DataFrame(fit(df), index = [0])

barr_nomenclature(stats_df)

plot_fits(df, stats_df)