#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 12:29:02 2018

@author: ian
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols

import utils

#------------------------------------------------------------------------------
# Class init
#------------------------------------------------------------------------------
class random_error(object):
    
    """ 
    Random error class
    
    Args:
        * dataframe (pandas dataframe): with columns containing required data
          (minimum of: turbulent flux, temperature, wind speed, insolation)
    
    Kwargs:
        * names_dict (dict): a dictionary containing the required 
          configuration items (default uses OzFluxQC nomenclature and is 
          compatible with a standard L5 dataset)
          (random_error.get_configs_dict())
        * num_bins (int): number of bins to use for the averaging of the errors
        * noct_threshold (int or float): the threshold (in Wm-2 insolation) below which
          the onset of night occurs
        * t_threshold (int or float): the difference threshold for temperature
        * ws_threshold (int or float): the difference threshold for wind_speed
        * k_threshold (int or float): the difference threshold for insolation
    """
    
    def __init__(self, dataframe, names_dict = False, num_bins = 50,
                 noct_threshold = 10, scaling_coefficient = 1,
                 t_threshold = 3, ws_threshold = 1, k_threshold = 35):
        
        if not names_dict: 
            self.external_names = self._define_default_external_names()
        else:
            self.external_names = names_dict
        self.internal_names = self._define_default_internal_names()
        self.df = utils.rename_df(dataframe, self.external_names, 
                                  self.internal_names)
        self._QC()
        self.num_bins = num_bins
        self.noct_threshold = noct_threshold
        self.scaling_coefficient = scaling_coefficient
        self.t_threshold = t_threshold
        self.ws_threshold = ws_threshold
        self.k_threshold = k_threshold
        self.binned_error = self.get_flux_binned_sigma_delta()
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Class methods
#------------------------------------------------------------------------------
        
    #--------------------------------------------------------------------------
    def _QC(self):
        
        interval = int(filter(lambda x: x.isdigit(), 
                              pd.infer_freq(self.df.index)))
        assert interval % 30 == 0
        recs_per_day = 1440 / interval
        self.recs_per_day = recs_per_day

        return
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def _define_default_external_names(self):
              
        return {'flux_name': 'Fc',
                'mean_flux_name': 'Fc_SOLO',
                'windspeed_name': 'Ws',
                'temperature_name': 'Ta',
                'insolation_name': 'Fsd'}
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def _define_default_internal_names(self):

        return {'flux_name': 'flux',
                'mean_flux_name': 'flux_mean',
                'windspeed_name': 'Ws',
                'temperature_name': 'Ta',
                'insolation_name': 'Fsd'}
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def get_flux_binned_sigma_delta(self):    

        """
        Calculate the daily differences and bin average as a function of 
        flux magnitude
        
        Returns:
            * Dictionary containing keys of 'night' and 'day', each of which
              contains a pandas dataframe with estimates of binned mean flux
              and corresponding sigma_delta estimate.
        """
        
        #----------------------------------------------------------------------
        # Bin day and night data
        def bin_time_series():
            
            def get_sigmas(df):
                def calc(s):
                    return abs(s - s.mean()).mean() * np.sqrt(2)
                return pd.DataFrame({'sigma_delta': 
                                      map(lambda x: 
                                          calc(df.loc[df['quantile_label'] == x, 
                                                      'flux_diff']), 
                                          df['quantile_label'].unique()
                                          .categories),
                                     'mean': 
                                      map(lambda x: 
                                          df.loc[df['quantile_label'] == x,
                                                 'flux_mean'].mean(),
                                          df['quantile_label'].unique()
                                          .categories)})

            noct_df = filter_df.loc[filter_df.Fsd_mean < self.noct_threshold, 
                                    ['flux_mean', 'flux_diff']]
            day_df = filter_df.loc[filter_df.Fsd_mean > self.noct_threshold, 
                                   ['flux_mean', 'flux_diff']]
                    
            nocturnal_propn = float(len(noct_df)) / len(filter_df)
            num_cats_night = int(round(self.num_bins * nocturnal_propn))
            num_cats_day = self.num_bins - num_cats_night
            
            noct_df['quantile_label'] = pd.qcut(noct_df.flux_mean, num_cats_night, 
                                                labels = np.arange(num_cats_night))
            noct_group_df = get_sigmas(noct_df)
    
            day_df['quantile_label'] = pd.qcut(day_df.flux_mean, num_cats_day, 
                                               labels = np.arange(num_cats_day))
            day_group_df = get_sigmas(day_df)

            return day_group_df, noct_group_df
        #----------------------------------------------------------------------

        #----------------------------------------------------------------------    
        def difference_time_series():
            diff_df = pd.DataFrame(index = self.df.index)
            for var in ['flux', 'Ta', 'Fsd', 'Ws']:
                var_name = var + '_diff'
                temp = self.df[var] - self.df[var].shift(self.recs_per_day) 
                diff_df[var_name] = temp if var == 'flux' else abs(temp)
            diff_df['flux_mean'] = (self.df['flux_mean'] + self.df['flux_mean']
                                    .shift(self.recs_per_day)) / 2
            diff_df['Fsd_mean'] = (self.df['Fsd'] + 
                                   self.df['Fsd'].shift(self.recs_per_day)) / 2
            return diff_df
        #----------------------------------------------------------------------
        
        #----------------------------------------------------------------------
        def filter_time_series():
            bool_s = ((diff_df['Ws_diff'] < self.ws_threshold) & 
                      (diff_df['Ta_diff'] < self.t_threshold) & 
                      (diff_df['Fsd_diff'] < self.k_threshold))
            return pd.DataFrame({var: diff_df[var][bool_s] for var in 
                                 ['flux_diff', 'flux_mean', 
                                  'Fsd_mean']}).dropna()
        #----------------------------------------------------------------------
        
        #----------------------------------------------------------------------
        # Main routine
        #----------------------------------------------------------------------

        diff_df = difference_time_series()
        filter_df = filter_time_series()
        day_df, noct_df = bin_time_series()
        return {'day': day_df, 'night': noct_df}
    
    #-------------------------------------------------------------------------- 
    
    #--------------------------------------------------------------------------
    def estimate_random_error(self):
        
        """ Generate single realisation of random error for time series """
        
        sigma_delta_series = self.estimate_sigma_delta()
        return pd.Series(np.random.laplace(0, sigma_delta_series / np.sqrt(2)),
                         index = sigma_delta_series.index)
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def estimate_sigma_delta(self):
        
        """Calculates sigma_delta value for each member of a time series"""
        
        work_df = self.df.copy()
        stats_dict = self.get_regression_statistics()
        work_df.loc[np.isnan(work_df.flux), 'flux_mean'] = np.nan
        work_df['sigma_delta'] = np.nan
        night_stats = stats_dict['night']['stats']
        work_df.loc[work_df.Fsd < self.noct_threshold, 'sigma_delta'] = (
            work_df.flux_mean * night_stats.slope + night_stats.intercept)
        day_stats = stats_dict['day']['stats']
        work_df.loc[work_df.Fsd > self.noct_threshold, 'sigma_delta'] = (
            work_df.flux_mean * day_stats.slope + day_stats.intercept)
        if any(work_df['sigma_delta'] < 0):
            n_below = len(work_df['sigma_delta'][work_df['sigma_delta'] < 0])
            print ('Warning: approximately {0} estimates of sigma_delta have '
                   'value less than 0 - setting to mean of all other values'
                   .format(str(n_below)))
            work_df.loc[work_df['sigma_delta'] < 0, 'sigma_delta'] = (
                work_df['sigma_delta']).mean()
        return work_df['sigma_delta']
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    # Calculate basic regression statistics
    def get_regression_statistics(self):
        
        regression_dict = {}
        data_dict = self.get_flux_binned_sigma_delta()
        for state in data_dict:
            df = data_dict[state].copy()
            regression = ols("data ~ x", data = dict(data = df['sigma_delta'], 
                                                     x = df['mean'])).fit()
            df = df.join(regression.outlier_test())
            outlier_list = df.loc[df['bonf(p)'] < 0.5].index.tolist()
            df = df.loc[df['bonf(p)'] > 0.5]
            statistics = stats.linregress(df['mean'], df['sigma_delta'])
            regression_dict[state] = {'stats': statistics}
            if not len(outlier_list) == 0: 
                regression_dict[state]['outliers'] = outlier_list
        return regression_dict
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def plot_data(self, flux_units = '\mu mol\/CO_2\/m^{-2}\/s^{-1}'):
        
        data_dict = self.binned_error
        stats_dict = self.get_regression_statistics()
        
        colour_dict = {'day': 'C1', 'night': 'C0'}
        
        x_min = min(map(lambda x: data_dict[x]['mean'].min(), 
                        data_dict.keys()))
        x_max = max(map(lambda x: data_dict[x]['mean'].max(), 
                        data_dict.keys()))
        y_max = max(map(lambda x: data_dict[x]['sigma_delta'].max(), 
                        data_dict.keys()))
        
        fig, ax1 = plt.subplots(1, 1, figsize = (14, 8))
        fig.patch.set_facecolor('white')
        ax1.xaxis.set_ticks_position('bottom')
        ax1.set_xlim([round(x_min * 1.05), round(x_max * 1.05)])
        ax1.set_ylim([0, round(y_max * 1.05)])
        ax1.yaxis.set_ticks_position('left')
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.tick_params(axis = 'y', labelsize = 14)
        ax1.tick_params(axis = 'x', labelsize = 14)
        ax1.set_xlabel('$flux\/({})$'.format(flux_units), fontsize = 18)
        ax1.set_ylabel('$\sigma[\delta]\/({})$'.format(flux_units), 
                       fontsize = 18)
        ax2 = ax1.twinx()
        ax2.spines['right'].set_position('zero')
        ax2.spines['left'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.set_ylim(ax1.get_ylim())
        ax2.tick_params(axis = 'y', labelsize = 14)
        plt.setp(ax2.get_yticklabels()[0], visible = False)
        outlier_df_list = []
        for state in data_dict.keys():
            stats = stats_dict[state]['stats']
            df = data_dict[state]
            if 'outliers' in stats_dict[state]:
                outlier_df_list.append(df.loc[stats_dict[state]['outliers']])
            x = np.linspace(df['mean'].min(), df['mean'].max(), 2)
            y = x * stats.slope + stats.intercept
            text_str = ('${0}: a = {1}, b = {2}, r^2 = {3}$'
                        .format(state[0].upper() + state[1:],
                        str(round(stats.slope, 2)),
                        str(round(stats.intercept, 2)),
                        str(round(stats.rvalue ** 2, 2))))
            ax1.plot(df['mean'], df['sigma_delta'], 'o', 
                     mfc = colour_dict[state], mec = 'black', label = text_str)
            ax1.plot(x, y, color = colour_dict[state])
        try:
            outlier_df = pd.concat(outlier_df_list)
            ax1.plot(outlier_df['mean'], outlier_df['sigma_delta'], 's', 
                     mfc = 'None', mec = 'black', ms = 15, 
                     label = '$Outlier\/(excluded)$')
        except ValueError:
            pass
        ax1.legend(loc = [0.05, 0.1], fontsize = 14)
        return fig
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def propagate_random_error(self, n_trials):
        
        """ Run Monte Carlo-style trials to assess uncertainty due to 
        random error over entire dataset
        
        Args:
            * n_trials (int):  number of trials over which to compound the sum.
            * scaling_coefficient (int or float): scales summed value to required \
            units.
        
        Returns:
            * float: scaled estimate of 2-sigma bounds of all random error\
            trial sums.
        """
        
        sigma_delta_series = self.estimate_sigma_delta()
        crit_t = stats.t.isf(0.025, n_trials)  
        results_list = []
        for this_trial in xrange(n_trials):
            results_list.append(pd.Series(np.random.laplace(0, 
                                                            sigma_delta_series / 
                                                            np.sqrt(2))).sum() *
                                self.scaling_coefficient)
        return round(float(pd.DataFrame(results_list).std() * crit_t), 2)
    #--------------------------------------------------------------------------