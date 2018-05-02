# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 14:37:38 2015

@author: imchugh
"""
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import utils
reload(utils)

# Note that day and night sampling not yet implemented!!!

class model_error(object):
    '''
    Class that calculates model error by testing the effect of removing the
    same proportion of data as is missing from the entire sample from 
    multiple subsamples
    
    Args:
        * dataframe (pandas dataframe): with columns containing required data
          (minimum of: observed and modelled fluxes)
    Kwargs:
        * scaling_coefficient (int or float): scaling required to calculate
          a specific summed quantity from the raw quantity - for example,
          to calculate gC m-2 over a given period, if the raw units are 
          umolC m-2 s-1, then the scaling required to convert is 
          12 (conversion from mol to gram) * 10^-6 (conversion from umol to
          mol)
        * minimum_pct (int or float): the minimum percentage of available 
          observational data required to proceed
        * names_dict (dict): a dictionary that maps the external data names to 
          the required variables (default uses OzFluxQC nomenclature and is 
          compatible with a standard L5 dataset - use this as a template to 
          create an alternative dictionary to pass; if names dict is None, 
          default is used)        
    '''
    def __init__(self, dataframe, scaling_coefficient = 1, minimum_pct = 20,
                 noct_threshold = 10, names_dict = None):

        if names_dict:
            self.external_names = names_dict
        else:
            self.external_names = self._define_default_external_names()
        self.internal_names = self._define_default_internal_names()
        self.df = utils.rename_df(dataframe, self.external_names, 
                                  self.internal_names)
        self.scaling_coefficient = scaling_coefficient
        self.minimum_pct = minimum_pct
        self._get_stats_and_qc()

    #--------------------------------------------------------------------------
    def best_estimate(self):
        
        '''
        Calculate the best estimate sum over the entire period
        '''
        
        return (self.df.Observations.where(~np.isnan(self.df.Observations), 
                                           self.df.Model).sum() *
                self.interval * 60 * self.scaling_coefficient)
    #--------------------------------------------------------------------------    
    
    #--------------------------------------------------------------------------
    def _define_default_external_names(self):

        return {'Observations': 'Fc',
                'Model': 'Fc_SOLO'}
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def _define_default_internal_names(self):

        return {'Observations': 'Observations',
                'Model': 'Model'}
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    
    '''
    Estimate the nmodel error in the population by subsampling
    
    Kwargs:
        * subsample_n (int): the size of the subsample to be used
    '''
    
    def estimate_model_error(self, subsample_n = 1000):
        
        sub_df = self.df.dropna()
        retain_obs_n = int(self.pct_available / 100 * subsample_n)
        random_index = np.random.randint(0, len(sub_df), subsample_n)
        temp_df = sub_df.iloc[random_index]
        observational_sum = (temp_df.Observations.sum() * self.interval * 60 *
                             self.scaling_coefficient)
        spliced_sum = (pd.concat([temp_df['Observations'].iloc[: retain_obs_n], 
                                  temp_df['Model'].iloc[retain_obs_n:]]).sum() * 
                       self.interval * 60 * self.scaling_coefficient)
        return (observational_sum - spliced_sum) / observational_sum * 100
    #--------------------------------------------------------------------------
    
    #--------------------------------------------------------------------------
    def _get_stats_and_qc(self):
        
        interval = int(filter(lambda x: x.isdigit(), 
                              pd.infer_freq(self.df.index)))
        if not interval % 30 == 0:
            raise RuntimeError('Dataset datetime index is non-contiguous - '
                               'exiting')
        df_length = len(self.df)
        model_length = len(self.df.loc[pd.isnull(self.df.Model) == 0])
        obs_length = len(self.df.loc[pd.isnull(self.df.Observations) == 0])
        pct_available = obs_length / float(df_length) * 100
        if model_length != df_length:
            raise RuntimeError('{} missing values in model series... aborting'
                               .format(str(df_length - model_length)))
        if pct_available < self.minimum_pct:
            raise RuntimeError('Insufficient data to proceed (minimum % '
                               'set to {0}, encountered only {1}%)... '
                               'returning'
                               .format(str(self.minimum_pct), 
                                       round(str(pct_available), 1)))
        self.interval = interval
        self.pct_available = pct_available
        return
    #--------------------------------------------------------------------------
        
    #--------------------------------------------------------------------------
    def plot_pdf(self, n_trials = 1000, 
                 units = '$Uncertainty\/(gC\/m^{-2}\/a^{-1})$'):
        
        pct_error, trials = self.propagate_model_error(n_trials = n_trials,
                                                       return_trials = True)
        summed_estimate = self.best_estimate()
        trials = np.array(trials) / 100 * summed_estimate + summed_estimate

        x_lo = summed_estimate - stats.norm.ppf(.999) * trials.std()
        x_hi = summed_estimate + stats.norm.ppf(.999) * trials.std()
        x = np.linspace(x_lo, x_hi, 100)
        mu = summed_estimate
        sig = trials.std()
        y = mlab.normpdf(x, mu, sig) 
        crit_t = stats.t.isf(0.025, len(trials))

        fig, ax = plt.subplots(1, figsize = (12, 8))
        x_min = x_lo - abs(x_lo * 0.025)
        x_max = x_hi + abs(x_hi * 0.025)
        ax.set_xlim([x_min, x_max])
        ax.set_xlabel(units, fontsize = 16)
        ax.set_ylabel('Relative frequency', fontsize = 16)
        ax.tick_params(axis = 'x', labelsize = 14)
        ax.tick_params(axis = 'y', labelsize = 14)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')    
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.hist(trials, bins = 100, normed = True)
        ax.plot(x, y, color = 'red', linewidth = 2.5, label = 'Gaussian PDF')
        ax.axvline(mu, color = 'black', lw = 0.75)
        ax.axvline(mu - sig * crit_t, color = 'black', ls = '-.', lw = 0.75)
        ax.axvline(mu + sig * crit_t, color = 'black', ls = '-.', lw = 0.75)
        ax.text(0.05, 0.9, 
                '$\mu\/=\/{0}$\n$\sigma\/=\/{1}$'.format
                (str(round(mu, 1)), str(round(sig, 1))),
                transform = ax.transAxes, fontsize = 14)
        return fig

    #--------------------------------------------------------------------------
    
    '''
    Propagate the calculated model error over the entire period of the sample
    
    Kwargs:
        * n_trials (int): the number of trials to use to test cross-trial 
          uncertainty
        * return_trials (bool): whether to return the individual trial results,
          or just the 95%CI
    '''
    
    def propagate_model_error(self, n_trials = 1000, return_trials = False):
        
        crit_t = stats.t.isf(0.025, n_trials)
        error_list = []
        for this_trial in xrange(n_trials):
            error_list.append(self.estimate_model_error())
        if not return_trials:
            np.array(error_list).std() * crit_t
        else:
            return np.array(error_list).std() * crit_t, error_list
    #--------------------------------------------------------------------------