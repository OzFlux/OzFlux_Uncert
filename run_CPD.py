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

def data():
    
    df = pd.read_csv('/home/ian/Documents/Test/data.csv')
    df.drop('Unnamed: 0', axis = 1, inplace = True)
    return df

def plot_fits(temp_df, stats_df):
    
    if any(np.isnan(stats_df.cp_a) | np.isnan(stats_df.cp_b)):
        print 'Unsuccessful fits... exiting'
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
    ax.set_xlim([0, round(max(temp_df.ustar) * 1.05, 1)])
    ax.set_ylim([0, round(max(temp_df.Fc) * 1.05, 1)])
    plt.plot(temp_df['ustar'], temp_df['Fc_clean'], color = 'black')
    plt.plot(temp_df['ustar'], temp_df['Fc'], 'bo')
    plt.plot(temp_df['ustar'], yhat_b_series, color = 'red')   
    plt.plot(temp_df['ustar'], yhat_a_series, color = 'green')   
    ax.set_xlabel('$u*\/(m\/s^{-1}$)', fontsize = 16)
    ax.set_ylabel('$F_c\/(\mu mol C\/m^{-2} s^{-1}$)', fontsize = 16)
    props = dict(boxstyle='round,pad=1', facecolor='white', alpha=0.5)
    txt='Change point detected at u*='+str(round(stats_df['ustar_th_b'],3))+' (i='+str(change_point_b_model)+')'
#    ax=plt.gca()
    plt.text(0.57,0.1,txt,bbox=props,fontsize=12,verticalalignment='top',transform=ax.transAxes)

noise_dict = {'slope': 0.44, 
              'intercept': 0.4}

intercept = 0
slope = 20
change_point = 15
n = 55
use_synth = False
noise_on = True


if use_synth:

    ustar_series = np.linspace(0, 0.6, n)
    
    flux_series = np.concatenate((ustar_series[:change_point] * slope + intercept,
                                  np.tile(ustar_series[change_point] * slope + intercept, 
                                          n - change_point)))
    
    sigma_delta_series = flux_series * noise_dict['slope'] + noise_dict['intercept']
    
    error_series = np.random.laplace(0, sigma_delta_series / np.sqrt(2))
    
    df = pd.DataFrame({'ustar': ustar_series, 'Fc_clean': flux_series})
    df['Fc'] = flux_series + error_series if noise_on else flux_series
    
else:
    
    df = data()
    
stats_df = pd.DataFrame(fit(df), index = [0])

test = plot_fits(df, stats_df)

print 'Significance level of a model is: {}'.format(str(round(stats_df.p_a, 5)))
print 'Significance level of b model is: {}'.format(str(round(stats_df.p_b, 5)))