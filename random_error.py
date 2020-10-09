#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 12:29:02 2018

@author: ian
"""

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
import pdb

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
        * names_dict (dict): a dictionary that maps the external data names to
          the required variables (default uses OzFluxQC nomenclature and is
          compatible with a standard L5 dataset - use this as a template to
          create an alternative dictionary to pass; if names dict is None,
          default is used)
        * num_bins (int): number of bins to use for the averaging of the errors
        * noct_threshold (int or float): the threshold (in Wm-2 insolation)
          below which the onset of night occurs
        * scaling_coefficient (int or float):
        * t_threshold (int or float): the difference threshold for temperature
        * ws_threshold (int or float): the difference threshold for wind_speed
        * k_threshold (int or float): the difference threshold for insolation
    """

    def __init__(self, dataframe, names_dict = False, num_bins = 50,
                 noct_threshold = 10, scaling_coefficient = 1,
                 t_threshold = 3, ws_threshold = 1, k_threshold = 35):

        if not names_dict:
            self.external_names = _define_default_external_names()
        else:
            self.external_names = names_dict
        self.internal_names = _define_default_internal_names()
        self.df = utils.rename_df(dataframe, self.external_names,
                                  self.internal_names)
        self.recs_per_day = _get_interval(self.df)
        self.num_bins = num_bins
        self.noct_threshold = noct_threshold
        self.scaling_coefficient = scaling_coefficient
        self.t_threshold = t_threshold
        self.ws_threshold = ws_threshold
        self.k_threshold = k_threshold
#------------------------------------------------------------------------------

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
            print(('Warning: approximately {0} estimates of sigma_delta have '
                   'value less than 0 - setting to mean of all other values'
                   .format(str(n_below))))
            work_df.loc[work_df['sigma_delta'] < 0, 'sigma_delta'] = (
                work_df['sigma_delta']).mean()
        return work_df['sigma_delta']
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def get_differenced_time_series(self, keep_met_diffs=False):

        """
        Returns the day-differenced time series with cases that fell outside
        meteorological constraints - or nans - dropped

        Args:
            * keep_met_diffs (bool): returns the differences in the
              meteorological constraints if true, otherwise they're dropped
        """

        diff_df = pd.DataFrame(index = self.df.index)
        for var in ['flux', 'Ta', 'Fsd', 'Ws']:
            var_name = var + '_diff'
            temp = self.df[var] - self.df[var].shift(self.recs_per_day)
            diff_df[var_name] = temp if var == 'flux' else abs(temp)
        diff_df['flux_mean'] = (self.df['flux_mean'] + self.df['flux_mean']
                                .shift(self.recs_per_day)) / 2
        diff_df['Fsd_mean'] = (self.df['Fsd'] +
                               self.df['Fsd'].shift(self.recs_per_day)) / 2
        keep_bool = ((diff_df['Ws_diff'] < self.ws_threshold) &
                     (diff_df['Ta_diff'] < self.t_threshold) &
                     (diff_df['Fsd_diff'] < self.k_threshold))
        diff_df = diff_df.loc[keep_bool].dropna()
        if keep_met_diffs: return diff_df
        return diff_df[['flux_diff', 'flux_mean', 'Fsd_mean']].dropna()
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

        # try: assert mode in ['day', 'night']
        # except AssertionError: raise KeyError('"Mode" kwarg must be either '
        #                                       '"day" or "night"')

        def calc_laplace_sd(s): return abs(s - s.mean()).mean() * np.sqrt(2)

        def get_sigmas(this_df):

            func = calc_laplace_sd
            sig_del = (
                [func(this_df.loc[this_df['quantile_label'] == x, 'flux_diff'])
                 for x in this_df['quantile_label'].unique().categories])
            mean = [this_df.loc[this_df['quantile_label'] == x, 'flux_mean'].mean()
                    for x in this_df['quantile_label'].unique().categories]
            return pd.DataFrame({'sigma_delta': sig_del, 'mean': mean})

        # if mode == 'day':
        df = self.get_differenced_time_series()

        noct_df = df.loc[df.Fsd_mean < self.noct_threshold,
                         ['flux_mean', 'flux_diff']]
        day_df = df.loc[df.Fsd_mean > self.noct_threshold,
                        ['flux_mean', 'flux_diff']]

        nocturnal_propn = float(len(noct_df)) / len(df)
        num_cats_night = int(round(self.num_bins * nocturnal_propn))
        num_cats_day = self.num_bins - num_cats_night

        noct_df['quantile_label'] = pd.qcut(noct_df.flux_mean, num_cats_night,
                                            labels = np.arange(num_cats_night))
        noct_group_df = get_sigmas(noct_df)

        day_df['quantile_label'] = pd.qcut(day_df.flux_mean, num_cats_day,
                                           labels = np.arange(num_cats_day))
        day_group_df = get_sigmas(day_df)

        return {'day': day_group_df, 'night': noct_group_df}
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

    def plot_histogram(self):

        def myround(x,base=10):
            return int(base*round(x/base))

        # Instantiate plot
        fig, ax1 = plt.subplots(1, 1, figsize = (14, 8))
        fig.patch.set_facecolor('white')

        # Calculate scaling parameter for Laplace (sigma / sqrt(2)) and Gaussian
        # (sigma) distributions over entire dataset
        df = self.get_differenced_time_series()
        beta = (abs(df.flux_diff - df.flux_diff.mean())).mean()
        sig = df.flux_diff.std()

        # Get edge quantiles and range, then calculate Laplace pdf over range
        x_low = myround(np.percentile(df.flux_diff, 0.5))
        x_high = myround(np.percentile(df.flux_diff, 99.5))
        x_range = (x_high - x_low)
        x = np.arange(x_low, x_high, 1 / (x_range * 10.))
        pdf_laplace = np.exp(-abs(x / beta)) / (2. * beta)
        pdf_gaussian = stats.norm(loc=0, scale=sig).pdf(x)

        # Plot normalised histogram with Laplacian and Gaussian pdfs
        ax1.hist(np.array(df.flux_diff), bins = 200,
                 range = [x_low, x_high], density=True, color = '0.7',
                 edgecolor = 'none')
        ax1.plot(x,pdf_laplace,color='black', label='Laplacian')
        ax1.plot(x,pdf_gaussian,color='black', linestyle = '--',
                label='Gaussian')
        ax1.set_xlabel(r'$\delta\/(\mu mol\/m^{-2} s^{-1}$)',fontsize=18)
        ax1.set_ylabel('$Frequency$', fontsize=18)
        ax1.axvline(x=0, color = 'black', lw=0.5)
        ax1.tick_params(axis = 'x', labelsize = 14)
        ax1.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
        ax1.tick_params(axis = 'y', labelsize = 14)
        ax1.legend(frameon=False)
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def plot_sigma_delta(self, flux_units = '\mu mol\/CO_2\/m^{-2}\/s^{-1}'):

        data_dict = self.get_flux_binned_sigma_delta()
        stats_dict = self.get_regression_statistics()

        colour_dict = {'day': 'C1', 'night': 'C0'}

        x_min = min([data_dict[x]['mean'].min() for x in list(data_dict.keys())])
        x_max = max([data_dict[x]['mean'].max() for x in list(data_dict.keys())])
        y_max = max([data_dict[x]['sigma_delta'].max() for x in list(data_dict.keys())])

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
        for state in list(data_dict.keys()):
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
        for this_trial in range(n_trials):
            results_list.append(pd.Series(np.random.laplace(0,
                                                            sigma_delta_series /
                                                            np.sqrt(2))).sum() *
                                self.scaling_coefficient)
        return round(float(pd.DataFrame(results_list).std() * crit_t), 2)
    #--------------------------------------------------------------------------

#------------------------------------------------------------------------------
### FUNCTIONS ###
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _define_default_external_names():

    return {'flux_name': 'Fc',
            'mean_flux_name': 'Fc_SOLO',
            'windspeed_name': 'Ws',
            'temperature_name': 'Ta',
            'insolation_name': 'Fsd'}
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _define_default_internal_names():

    return {'flux_name': 'flux',
            'mean_flux_name': 'flux_mean',
            'windspeed_name': 'Ws',
            'temperature_name': 'Ta',
            'insolation_name': 'Fsd'}
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _get_interval(df):

    interval = int(''.join([x for x in pd.infer_freq(df.index) if x.isdigit()]))
    assert interval % 30 == 0
    return int(1440 / interval)
#------------------------------------------------------------------------------