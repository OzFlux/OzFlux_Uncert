# -*- coding: utf-8 -*-

# Python modules
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import pandas as pd
from scipy import stats
import os
from scipy.interpolate import PchipInterpolator

import utils

#------------------------------------------------------------------------------
# Class init
#------------------------------------------------------------------------------
class change_point_detect(object):
    '''
    Put docstring in here
    '''
    def __init__(self, dataframe, resample = True, names_dict = False, 
                 insolation_threshold = 10, season_routine = 'standard'):

        interval = int(filter(lambda x: x.isdigit(), 
                              pd.infer_freq(dataframe.index)))
        assert interval % 30 == 0
        assert season_routine in ['standard', 'barr']
        if not names_dict: 
            self.external_names = self._define_default_external_names()
        else:
            self.external_names = names_dict
        self.df = utils.rename_df(dataframe, self.external_names, 
                                  self._define_default_internal_names())
        self.resample = resample
        self.insolation_threshold = insolation_threshold
        self.season_routine = season_routine
        self.interval = interval
        self.season_n = 1000 if interval == 30 else 600
        self.bin_n = 5 if interval == 30 else 3
        self.valid_years_list = self._get_valid_years()
#------------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def _cross_sample_stats_QC(self, df):
        
        year = np.unique(df.index.get_level_values(0)).item()
        d_mode = len(df.loc[df.b1 > 0, 'b1'])
        e_mode = len(df.loc[df.b1 < 0, 'b1'])
        if e_mode > d_mode:
            df.loc[df.b1 > 0, ['ustar_th_b', 'b0', 'b1']] = np.nan
        else:
            df.loc[df.b1 < 0, ['ustar_th_b', 'b0', 'b1']] = np.nan
        valid_n = df.ustar_th_b.count()
        stats_df = pd.DataFrame({'norm_a1': (df.a1 * (df.ustar_th_a / 
                                                      (df.a0 + df.a1 * 
                                                       df.ustar_th_a))).median(),
                                 'norm_a2': (df.a2 * (df.ustar_th_a / 
                                                      (df.a0 + df.a1 * 
                                                       df.ustar_th_a))).median(),
                                 'ustar_mean': df.ustar_th_b.mean(),
                                 'ustar_sig': df.ustar_th_b.std(),
                                 'ustar_valid_n': valid_n,
                                 'crit_t_value': stats.t.isf(0.025, valid_n)},
                                index = [year])
        df = df[['b0', 'b1', 'ustar_th_b']].dropna()
        df.index = np.tile(year, len(df))
        return {'trial_results': df, 'summary_statistics': stats_df}
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def _define_default_external_names(self):
              
        return {'flux_name': 'Fc',
                'temperature_name': 'Ta',
                'insolation_name': 'Fsd',
                'friction_velocity_name': 'ustar'}
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def _define_default_internal_names(self):

        return {'flux_name': 'NEE',
                'temperature_name': 'Ta',
                'insolation_name': 'Fsd',
                'friction_velocity_name': 'ustar'}
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def get_change_points(self, n_trials = 1, write_to_dir = None,
                          keep_trial_results = False):

        stats_lst = []
        trials_lst = []
        print 'Getting change points for year:'        
        for year in self.valid_years_list:
            print '    {}'.format(str(year)),
            results_dict = self.get_change_points_for_year(year, n_trials)
            if results_dict:
                stats_lst.append(results_dict['summary_statistics'])
                trials_lst.append(results_dict['trial_results'])
        if not stats_lst:
            print 'Could not find any valid change points!'
            return
        output_dict = {'summary_statistics': pd.concat(stats_lst)}
        if keep_trial_results: 
            output_dict['trial_results'] = pd.concat(trials_lst)
        if write_to_dir: self._write_to_file(output_dict, write_to_dir)
        return output_dict
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def get_change_points_for_year(self, year, n_trials):
        
        if not self.resample:
            if not n_trials == 1:
                print ('Multiple trials without resampling are redundant! '
                       'Setting n_trials to 1...')
                n_trials = 1          
        data_list = []
        print '- running trial #',
        season_func = self._get_season_function()
        for trial in xrange(n_trials):
            print str(trial + 1),
            try: df = season_func(year)
            except RuntimeError: continue
            idx = df.groupby(['Year', 'Season', 'T_class']).mean().index
            results_df = pd.DataFrame(map(lambda x: 
                                          fit(df.loc[x]), idx),
                                      index = idx)
            data_list.append(results_df)
        print 'Done!'
        results_df = pd.concat(data_list)
        return self._cross_sample_stats_QC(results_df)
    #--------------------------------------------------------------------------
    
    #--------------------------------------------------------------------------
    def _get_valid_years(self):
        
        l = []
        for year in sorted(list(set(self.df.index.year))):
            try:
                self._get_sample_data(self.df.loc[str(year)])
                l.append(year)
            except RuntimeError:
                continue
        return l
    #--------------------------------------------------------------------------
    
    #--------------------------------------------------------------------------
    def _get_sample_data(self, df):
        
        temp_df = df.loc[df['Fsd'] < self.insolation_threshold, 
                         ['NEE', 'ustar', 'Ta']].dropna()
        temp_df = temp_df[(temp_df.ustar >= 0) & (temp_df.ustar <= 3)]
        if len(temp_df) == 0:
            raise RuntimeError('No data available')
        if self.resample:
            temp_df = temp_df.iloc[sorted(np.random.randint(0, 
                                                            len(temp_df) - 1, 
                                                            len(temp_df)))]
        if not len(temp_df) > 4 * self.season_n: 
            raise RuntimeError('Insufficent data available')
        return temp_df
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def get_season_data(self, year = None):

        # Extract overlapping series to individual dataframes, for each of 
        # which: # 1) sort by temperature; 2) create temperature class; 
        # 3) sort temperature class by u*; 4) add bin numbers to each class, 
        # then; 5) concatenate
        years_lst = []
        if year: 
            assert isinstance(year, int)
            assert year in self.valid_years_list
            years = [year]
        else:
            years = self.valid_years_list
        for year in years:
            df = self._get_sample_data(self.df.loc[str(year)])
            df['Year'] = year
            n_seasons = len(df) / (self.season_n / 2) - 1
            T_array = np.concatenate(map(lambda x: np.tile(x, self.season_n / 4), 
                                         range(4)))
            bin_array = np.tile(np.concatenate(map(lambda x: np.tile(x, self.bin_n), 
                                                   range(50))), 4)
            seasons_lst = []
            for season in xrange(n_seasons):
                start_ind = season * (self.season_n / 2)
                end_ind = season * (self.season_n / 2) + self.season_n
                this_df = df.iloc[start_ind: end_ind].copy()
                this_df.sort_values('Ta', axis = 0, inplace = True)
                this_df['Season'] = season + 1
                this_df['T_class'] = T_array
                this_df = pd.concat(map(lambda x: 
                                        this_df.loc[this_df.T_class == x]
                                        .sort_values('ustar', axis = 0), 
                                        range(4)))
                this_df['Bin'] = bin_array
                seasons_lst.append(this_df)
            seasons_df = pd.concat(seasons_lst)
        
            # Construct multiindex and use Season, T_class and Bin as levels,
            # drop them as df variables then average by bin and drop it from the 
            # index
            arrays = [seasons_df.Year.values, seasons_df.Season.values, 
                      seasons_df.T_class.values, seasons_df.Bin.values]
            name_list = ['Year', 'Season', 'T_class', 'Bin']
            tuples = list(zip(*arrays))
            hierarchical_index = pd.MultiIndex.from_tuples(tuples, 
                                                           names = name_list)          
            seasons_df.index = hierarchical_index
            seasons_df.drop(name_list, axis = 1, inplace = True)
            seasons_df = seasons_df.groupby(level = name_list).mean()
            seasons_df.reset_index(level = ['Bin'], drop = True, inplace = True)
            years_lst.append(seasons_df)
        return pd.concat(years_lst)        
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def get_season_data_barrlike(self, year = None):

        # Extract overlapping series to individual dataframes, for each of 
        # which: # 1) sort by temperature; 2) create temperature class; 
        # 3) sort temperature class by u*; 4) add bin numbers to each class, 
        # then; 5) concatenate
        years_lst = []
        if year: 
            assert isinstance(year, int)
            assert year in self.valid_years_list
            years = [year]
        else:
            years = self.valid_years_list
        for year in years:
            df = self._get_sample_data(self.df.loc[str(year)])
            df['Year'] = year            
            seasons_lst = []
            df = pd.concat([df.loc[df.index.dayofyear >= 336], 
                            df.loc[df.index.dayofyear < 336]])
            n_seasons = len(df) / self.season_n
            n_per_season = (len(df) / (n_seasons * self.bin_n * 4)  
                            * self.bin_n * 4)
            n_per_Tclass = n_per_season / 4
            n_bins = n_per_Tclass / self.bin_n
            T_array = np.concatenate(map(lambda x: np.tile(x, n_per_Tclass), 
                                         range(4)))
            bin_array = np.tile(np.concatenate(map(lambda x: np.tile(x, self.bin_n), 
                                                   range(n_bins))), 4)
            for season in xrange(n_seasons):
                start_ind = season * n_per_season
                end_ind = start_ind + n_per_season
                this_df = df.iloc[start_ind: end_ind].copy()
                this_df.sort_values('Ta', axis = 0, inplace = True)
                this_df['Season'] = season + 1
                this_df['T_class'] = T_array
                this_df = pd.concat(map(lambda x: 
                                        this_df.loc[this_df.T_class == x]
                                        .sort_values('ustar', axis = 0), 
                                        range(4)))
                this_df['Bin'] = bin_array
                seasons_lst.append(this_df)
            seasons_df = pd.concat(seasons_lst)
        
            # Construct multiindex and use Season, T_class and Bin as levels,
            # drop them as df variables then average by bin and drop it from the 
            # index
            arrays = [seasons_df.Year, seasons_df.Season.values, 
                      seasons_df.T_class.values, seasons_df.Bin.values]
            name_list = ['Year', 'Season', 'T_class', 'Bin']
            tuples = list(zip(*arrays))
            hierarchical_index = pd.MultiIndex.from_tuples(tuples, 
                                                           names = name_list)          
            seasons_df.index = hierarchical_index
            seasons_df.drop(name_list, axis = 1, inplace = True)
            seasons_df = seasons_df.groupby(level = name_list).mean()
            seasons_df.reset_index(level = ['Bin'], drop = True, inplace = True)
            years_lst.append(seasons_df)
        return pd.concat(years_lst)    
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def _get_season_function(self):
        d = {'standard': self.get_season_data,
             'barr': self.get_season_data_barrlike}
        return d[self.season_routine]
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def _write_to_file(self, d, path_to_dir):
        
        if not os.path.isdir(path_to_dir): os.mkdir(path_to_dir)
        path_file = os.path.join(path_to_dir, 'change_points.xlsx')
        xlwriter = pd.ExcelWriter(path_file)
        for key in sorted(d.keys()):
            d[key].to_excel(xlwriter, sheet_name = key)
    #--------------------------------------------------------------------------
    
#------------------------------------------------------------------------------
def fit(sample_df):
    
    def a_model_statistics(cp):
        work_df = sample_df.copy()
        work_df['ustar_a1'] = work_df['ustar']
        work_df['ustar_a1'].iloc[cp + 1:] = work_df['ustar_a1'].iloc[cp]
        dummy_array = np.concatenate([np.zeros(cp + 1), 
                                      np.ones(df_length - (cp + 1))])
        work_df['ustar_a2'] = (work_df['ustar'] - 
                               work_df['ustar'].iloc[cp]) * dummy_array
        reg_params = np.linalg.lstsq(work_df[['int','ustar_a1','ustar_a2']],
                                     work_df['NEE'], rcond = None)[0]
        yHat = (reg_params[0] + reg_params[1] * work_df['ustar_a1'] +
                reg_params[2] * work_df['ustar_a2'])
        SSE_full = ((work_df['NEE'] - yHat)**2).sum()
        f_score = (SSE_null_a - SSE_full) / (SSE_full / (df_length - 3))
        return f_score, reg_params
    
    def b_model_statistics(cp):
        work_df = sample_df.copy()
        work_df['ustar_b'] = work_df['ustar']
        work_df['ustar_b'].iloc[cp + 1:] = work_df['ustar_b'].iloc[cp]
        reg_params = np.linalg.lstsq(work_df[['int','ustar_b']], 
                                     work_df['NEE'], rcond = None)[0]
        yHat = reg_params[0] + reg_params[1] * work_df['ustar_b']
        SSE_full = ((work_df['NEE'] - yHat)**2).sum()
        f_score = (SSE_null_b - SSE_full) / (SSE_full / (df_length - 2))
        return f_score, reg_params
    
    # Get stuff ready
    sample_df = sample_df.reset_index(drop = True)
    sample_df = sample_df.astype(np.float64)        
    df_length = len(sample_df)
    endpts_threshold = int(np.floor(df_length * 0.05))
    if endpts_threshold < 3: endpts_threshold = 3
    psig = 0.05
    
    # Calculate null model SSE for operational (b) and diagnostic (a) model
    SSE_null_b = ((sample_df['NEE'] - sample_df['NEE'].mean())**2).sum()
    alpha0 , alpha1 = stats.linregress(sample_df['ustar'], 
                                       sample_df['NEE'])[:2]
    SSE_null_a = ((sample_df['NEE'] - (sample_df['ustar'] * 
                                      alpha0 + alpha1))**2).sum()
    
    # Create arrays to hold statistics
    f_a_array = np.zeros(df_length)
    f_b_array = np.zeros(df_length)
    
    # Add series to df for numpy linalg
    sample_df['int'] = np.ones(df_length)
        
    # Iterate through all possible change points 
    for i in xrange(endpts_threshold, df_length - endpts_threshold):
                  
        # Diagnostic (a) and operational (b) model statistics
        f_a_array[i] = a_model_statistics(i)[0]
        f_b_array[i] = b_model_statistics(i)[0]

    # Get max f-score, associated change point and ustar value for models
    # (conditional on passing f score)
    d = {}
    fmax_a, cp_a = f_a_array.max(), int(f_a_array.argmax())
    p_a = f_test(fmax_a, df_length, model = 'a')    
    if p_a < psig:
        d['ustar_th_a'] = sample_df['ustar'].iloc[cp_a]
        d['a0'], d['a1'], d['a2'] = a_model_statistics(cp_a)[1] 
    else:
        for var in ['ustar_th_a', 'a0', 'a1', 'a2']:
            d[var] = np.nan
                      
    fmax_b, cp_b = f_b_array.max(), int(f_b_array.argmax())
    p_b = f_test(fmax_b, len(sample_df), model = 'b')
    if p_b < psig:    
        d['ustar_th_b'] = sample_df['ustar'].iloc[cp_b]
        d['b0'], d['b1'] = b_model_statistics(cp_b)[1]
    else:
        for var in ['ustar_th_b', 'b0', 'b1']:
            d[var] = np.nan

    return d
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#def f_test(self, f_max, n, model):
def f_test(f_max, n, model):
    
    p = np.NaN
    assert ~np.isnan(f_max)
    assert ~np.isnan(n)
    assert n > 10
    assert model == 'a' or model == 'b'
    
    if model == 'b':
        
        arr = np.array([[3.9293, 6.2992, 9.1471, 18.2659], 
                        [3.7734, 5.6988, 7.8770, 13.8100],
                        [3.7516, 5.5172, 7.4426, 12.6481],
                        [3.7538, 5.3224, 7.0306, 11.4461],
                        [3.7941, 5.3030, 6.8758, 10.6635],
                        [3.8548, 5.3480, 6.8883, 10.5026],
                        [3.9798, 5.4465, 6.9184, 10.4527],
                        [4.0732, 5.5235, 6.9811, 10.3859],
                        [4.1467, 5.6136, 7.0624, 10.5596],
                        [4.2770, 5.7391, 7.2005, 10.6871],
                        [4.4169, 5.8733, 7.3421, 10.6751],
                        [4.5556, 6.0591, 7.5627, 11.0072],
                        [4.7356, 6.2738, 7.7834, 11.2319]])
        idx = [10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000]
        cols = [0.8, 0.9, 0.95, 0.99]
        degfree = 2
        
    if model == 'a':
        
        arr = [[11.646, 15.559, 28.412],
               [9.651, 11.948, 18.043],
               [9.379, 11.396, 16.249],
               [9.261, 11.148, 15.75],
               [9.269, 11.068, 15.237],
               [9.296, 11.072, 15.252],
               [9.296, 11.059, 14.985],
               [9.341, 11.072, 15.013],
               [9.397, 11.08, 14.891],
               [9.398, 11.085, 14.874],
               [9.506, 11.127, 14.828],
               [9.694, 11.208, 14.898],
               [9.691, 11.31, 14.975],
               [9.79, 11.406, 14.998],
               [9.794, 11.392, 15.044],
               [9.84, 11.416, 14.98],
               [9.872, 11.474, 15.072],
               [9.929, 11.537, 15.115],
               [9.955, 11.552, 15.086],
               [9.995, 11.549, 15.164],
               [10.102, 11.673, 15.292],
               [10.169, 11.749, 15.154],
               [10.478, 12.064, 15.519]]
        idx = np.concatenate([np.linspace(10, 100, 10), 
                              np.linspace(150, 600, 10), 
                              np.array([800, 1000, 2500])])
        cols = [0.9, 0.95, 0.99]
        degfree = 3

    crit_table = pd.DataFrame(arr, index = idx, columns = cols)
    p_bounds = map(lambda x: 1 - (1 - x) / 2, [cols[0], cols[-1]])
    f_crit_vals = map(lambda x: float(PchipInterpolator(crit_table.index, 
                                                        crit_table[x])(n)), 
                      crit_table.columns)
    if f_max < f_crit_vals[0]:
        input_p = 1 - ((1 - p_bounds[0]) / 2)
        f_adj = (stats.f.ppf(input_p, degfree, n) 
                 * f_max / f_crit_vals[0])
        p = 2 * (1 - stats.f.cdf(f_adj, degfree, n))
        if p > 1: p = 1 
    elif f_max > f_crit_vals[-1]:
        input_p = 1 - ((1 - p_bounds[-1]) / 2)
        f_adj = (stats.f.ppf(input_p, degfree, n) 
                 * f_max / f_crit_vals[-1])
        p = 2 * (1 - stats.f.cdf(f_adj, degfree, n))
        if p < 0: p = 0
    else:
        p = PchipInterpolator(f_crit_vals, 
                              (1 - np.array(cols)).tolist())(f_max)
    return p
#------------------------------------------------------------------------------

#--------------------------------------------------------------------------
def plot_fit(df):
    
    plot_df = df.copy().reset_index(drop = True)
    stats_df = pd.DataFrame(fit(df), index = [0])
    if stats_df.empty:
        raise RuntimeError('Could not find a valid changepoint for this '
                           'sample')
    zero_list = [np.nan, 0, np.nan]
    if 'ustar_th_b' in stats_df:
        zero_list.append(stats_df.b0.item())
        cp_b = np.where(df.ustar == stats_df.ustar_th_b.item())[0].item()
        plot_df['yHat_b'] = (stats_df.ustar_th_b.item() * stats_df.b1.item() + 
                             stats_df.b0.item())
        plot_df['yHat_b'].iloc[:cp_b] = (plot_df.ustar.iloc[:cp_b] * 
                                         stats_df.b1.item() +
                                         stats_df.b0.item())
        
    if 'ustar_th_a' in stats_df:
        zero_list.append(stats_df.a0.item())
        cp_a = np.where(df.ustar == stats_df.ustar_th_a.item())[0].item()
        NEE_at_cp_a = (stats_df.ustar_th_a.item() * stats_df.a1.item() + 
                       stats_df.a0.item())
        if 'ustar_th_a' in stats_df:            
            plot_df['yHat_a'] = (plot_df.ustar * stats_df.a1.item() + 
                                 stats_df.a0.item())
            plot_df['yHat_a'].iloc[cp_a + 1:] = ((plot_df.ustar.iloc[cp_a + 1:] -
                                                  stats_df.ustar_th_a.item()) *
                                                 stats_df.a2.item() + 
                                                 NEE_at_cp_a)
    plot_df.loc[-1] = zero_list
    plot_df.index = plot_df.index + 1
    plot_df = plot_df.sort_index()
    fig, ax = plt.subplots(1, 1, figsize = (14, 8))
    ax.set_xlim([0, plot_df.ustar.max() * 1.05])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis = 'y', labelsize = 14)
    ax.tick_params(axis = 'x', labelsize = 14)
    fig.patch.set_facecolor('white')
    ax.set_xlabel('$u*\/(m\/s^{-1}$)', fontsize = 16)
    ax.set_ylabel('$NEE\/(\mu mol C\/m^{-2} s^{-1}$)', fontsize = 16)
    ax.axhline(0, color = 'black', lw = 0.5)
    ax.plot(plot_df.ustar, plot_df.NEE, 'bo', label = 'observational data')
    if 'ustar_th_b' in stats_df:
        ax.plot(plot_df.ustar, plot_df.yHat_b, color = 'red', 
                label = 'operational model')
    if 'ustar_th_a' in stats_df:
        ax.plot(plot_df.ustar, plot_df.yHat_a, color = 'green',
                label = 'diagnostic model')
    ax.legend(loc = (0.05, 0.85), fontsize = 12, frameon = False)
    return
#------------------------------------------------------------------------------

## Plot PDF of u* values and write to specified folder           
#def plot_hist(S,mu,sig,crit_t,year,plot_out):
#    S=S.reset_index(drop=True)
#    x_low=S.min()-0.1*S.min()
#    x_high=S.max()+0.1*S.max()
#    x=np.linspace(x_low,x_high,100)
#    fig=plt.figure(figsize=(12,8))
#    fig.patch.set_facecolor('white')
#    plt.hist(S,normed=True)
#    plt.plot(x,mlab.normpdf(x,mu,sig),color='red',linewidth=2.5,label='Gaussian PDF')
#    plt.xlim(x_low,x_high)
#    plt.xlabel(r'u* ($m\/s^{-1}$)',fontsize=16)
#    plt.axvline(x=mu-sig*crit_t,color='black',linestyle='--')
#    plt.axvline(x=mu+sig*crit_t,color='black',linestyle='--')
#    plt.axvline(x=mu,color='black',linestyle='dotted')
#    props = dict(boxstyle='round,pad=1', facecolor='white', alpha=0.5)
#    txt='mean u*='+str(mu)
#    ax=plt.gca()
#    plt.text(0.4,0.1,txt,bbox=props,fontsize=12,verticalalignment='top',transform=ax.transAxes)
#    plt.legend(loc='upper left')
#    plt.title(str(year)+'\n')
#    plot_out_name='ustar'+str(year)+'.jpg'
#    fig.savefig(os.path.join(plot_out,plot_out_name))
#    plt.close(fig)