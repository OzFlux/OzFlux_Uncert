# -*- coding: utf-8 -*-

# Python modules
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import pandas as pd
from scipy import stats
import os
from scipy.interpolate import PchipInterpolator
import xarray as xr
import pdb

import utils

#------------------------------------------------------------------------------
### CLASSES ###
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
class change_point_detect(object):

    """
    Class that determines change points for CO2 flux as function of u*

    Args:
        * dataframe (pandas dataframe): dataframe containing the data for
          analysis (must contain series of insolation, friction velocity,
          temperature and CO2 flux); note that NO qc of any kind is done -
          nan values are handled, but processing is otherwise naive
        * names_dict (python dict or None): dictionary containing the names of
          the required variables (see above) - must have the following
          structure: \n
          {'flux_name': <name>,
           'temperature_name': <name>,
           'insolation_name': <name>,
           'friction_velocity_name': <name>}
          If None is passed, the default dictionary is used for external names,
          as follows: \n
          {'flux_name': 'Fc',
           'temperature_name': 'Ta',
           'insolation_name': 'Fsd',
           'friction_velocity_name': 'ustar'}
        * insolation_threshold (int or float): threshold light level for
          filtering day and night conditions
        * minimum_annual_n (int): minimum number of valid records in a year for
          that year to be eligible for processing
    """

    def __init__(self, dataframe, missing_data=-9999, names_dict=None,
                 insolation_threshold=10, minimum_annual_n=None):

        interval = int(''.join(([x for x in pd.infer_freq(dataframe.index)
                                 if x.isdigit()])))
        try: assert interval % 30 == 0
        except AssertionError: raise RuntimeError(
                'Unrecognised or non-continuous dataframe DateTime index')
        if minimum_annual_n:
            try: assert isinstance(minimum_annual_n, int)
            except AssertionError: raise TypeError(
                'Kwarg "minimum_annual_n" must be of type int')
        if not names_dict:
            self.external_names = _define_default_external_names()
        else:
            self.external_names = names_dict
        self.df = utils.rename_df(dataframe, self.external_names,
                                  _define_default_internal_names())
        self.insolation_threshold = insolation_threshold
        self.interval = interval
        season_n = 1000 if interval == 30 else 600
        if not minimum_annual_n: self.minimum_annual_n = 4 * season_n
        self.missing_data=missing_data
#------------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def get_change_points(self, n_trials, years=None, resample=True,
                          do_cross_sample_qc=True, write_to_dir=None):

        """Get NEE change points as a function of ustar
           Args:
               * n_trials (int): number of trials to run
           Kwargs:
               * years (int, default None): years for which to run trials
                 (will parse all years if None is passed)
               * resample (boolean, default True): whether to randomly
                 resample data (note that if False, n_trials is forced to 1)
               * do_cross_sample_qc (boolean, default True): whether to return
                 cross_sample qc and resulting statistics,
                 or just raw fit data"""

        if not resample: n_trials = 1
        if write_to_dir: _check_path_exists(write_to_dir)
        df_list = []
        year_stats = self.get_season_stats(years=years, valid_years_only=True)
        if not year_stats: return
        print('Getting change points for year:')
        for year in year_stats.keys():
            print('    {}'.format(year), end=' ')
            trials_list = []
            print('- running trial #', end=' ')
            for trial in range(n_trials):
                char = ' ' if trial + 1 < n_trials else '\n'
                print(str(trial + 1), end=char)
                year_df = self.get_season_data(years=[year], resample=resample)
                cp_df = _fit_season_data(year_df)
                cp_df['bootstrap_n'] = trial
                trials_list.append(cp_df)
            results_df = pd.concat(trials_list).reset_index()
            results_df['Year'] = int(year)
            df_list.append(results_df)
        if len(df_list) == 0: return
        years_df = pd.concat(df_list)
        if do_cross_sample_qc:
            output_dict = _cross_sample_stats_QC(years_df)
            if write_to_dir: _write_to_file(output_dict, write_to_dir)
            return output_dict
        if write_to_dir:
            _write_to_file({'raw_results': years_df}, write_to_dir)
        return years_df
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def get_season_data(self, years=None, resample=True):

        """Get data sorted by year, season and temperature class, and
           bin-averaged
           Args:
               * years (list), default None: list of requested years
                 (data will only be returned for years with valid data);
                 default returns all eligible years
               * resample (boolean), default True: resample data
           Returns:
               * pandas dataframe (multi-indexed with Year, Season and
                 T class) containing ustar and NEE data
        """

        year_stats = self.get_season_stats(years=years, valid_years_only=True)
        if not year_stats: return
        df_list = []
        for this_year in year_stats.keys():
            str_year = str(this_year)
            this_df = self.get_valid_df().loc[str_year]
            if resample: this_df = get_resampled_data(this_df)
            season_df = _get_season_data(df=this_df,
                                         stats_dict=year_stats[str_year])
            season_df['Year'] = int(this_year)
            df_list.append(
                season_df.set_index('Year', append=True)
                .reorder_levels(['Year', 'Season', 'Tclass'])
                )
        return pd.concat(df_list)
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def get_season_stats(self, years=None, valid_years_only=False):

        """Get statistics of data availability for each year
           Args:
               * years (list), default None: list of requested years
                 (data will only be returned for years with valid data);
                 default returns all eligible years
               * valid_years_only (boolean), default False: whether to include
                 the statistical analysis results for years that either have no
                 or insufficient valid data
           Returns:
               * python dict (key: value pair of year and statistics for that
                 year)
        """

        data_years = self.df.index.year.unique().tolist()
        if years:
            try: assert isinstance(years, list)
            except AssertionError:
                raise TypeError("'years' parameter must be of type list")
            years_list = list(filter(lambda x: x in data_years,
                                     [int(y) for y in years]))
            if len(years_list) == 0:
                raise KeyError("None of years in list are contained in "
                               "supplied data")
        else:
            years_list = data_years
        valid_df = self.get_valid_df()
        stats_dict = {}
        no_stats_dict = {}
        for year in years_list:
            try:
                this_dict = (
                    _get_season_stats(valid_df.loc[str(year)], self.interval,
                                      self.minimum_annual_n)
                    )
                if  this_dict['sufficient_data'] == 'False':
                    no_stats_dict.append(this_dict)
                    print('Insufficient valid data for year {}'
                          .format(str(year)))
                    continue
                stats_dict[str(year)] = this_dict
            except KeyError:
                no_stats_dict[str(year)] = {'valid_data': 'False'}
                print('No valid data for year {}'.format(str(year)))
        if valid_years_only:
            if len(stats_dict) == 0: return
            return stats_dict
        stats_dict.update(no_stats_dict)
        return {x: stats_dict[x] for x in sorted(stats_dict.keys())}
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def get_valid_df(self):

        """Return a dataframe of all valid nocturnal data records"""

        return (
            self.df.loc[(self.df.Fsd < 10) &
                        (self.df.ustar < 3), ['NEE', 'Ta', 'ustar']]
            .replace(self.missing_data, np.nan)
            .dropna()
            )
    #--------------------------------------------------------------------------

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
class change_point_detect_from_netcdf(change_point_detect):

    def __init__(self, nc_path, missing_data=-9999, names_dict=None,
                 insolation_threshold=10, minimum_annual_n=None):

        dataset = xr.open_dataset(nc_path)
        dataframe = dataset.to_dataframe()
        dataset.close()
        dataframe.index = dataframe.index.droplevel([0, 1])
        interval = int(''.join(([x for x in pd.infer_freq(dataframe.index)
                                 if x.isdigit()])))
        try: assert interval % 30 == 0
        except AssertionError: raise RuntimeError(
                'Unrecognised or non-continuous dataframe DateTime index')
        if minimum_annual_n:
            try: assert isinstance(minimum_annual_n, int)
            except AssertionError: raise TypeError(
                'Kwarg "minimum_annual_n" must be of type int')
        if not names_dict:
            self.external_names = _define_default_external_names()
        else:
            self.external_names = names_dict
        self.df = utils.rename_df(dataframe, self.external_names,
                                  _define_default_internal_names())
        self.insolation_threshold = insolation_threshold
        self.interval = interval
        self.nc_path = nc_path
        season_n = 1000 if interval == 30 else 600
        if not minimum_annual_n: self.minimum_annual_n = 4 * season_n
        self.missing_data=missing_data
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    def write_change_points_to_nc(self, n_trials):

        df = self.get_change_points(n_trials)['summary_statistics']
        ustar_str = (
            ', '.join(
            [': '.join([str(x),
                        str(round(df.loc[df.Year==x, 'ustar_mean'].item(), 3))])
             for x in df.Year])
            )
        ds = xr.open_dataset(self.nc_path)
        ds.attrs['ustar_thresholds'] = ustar_str
        dir_path = os.path.dirname(self.nc_path)
        f_name = os.path.basename(self.nc_path)
        out_file = (
            os.path.join(dir_path,
                         f_name.split('.')[0] + '_ustar.' + f_name.split('.')[1])
            )
        return ds.to_netcdf(out_file, format='NETCDF4')
    #--------------------------------------------------------------------------

#------------------------------------------------------------------------------



#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
### FUNCTIONS ###
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _a_model(x, y, cp, null_SSE):

    x_a1 = np.concatenate([x[:cp], np.tile(x[cp], len(x) - cp)])
    dummy_array = np.concatenate([np.zeros(cp), np.ones(len(x) - cp)])
    x_a2 = (x - x[cp]) * dummy_array
    input_mtx = np.concatenate([np.expand_dims(np.ones(len(x)), axis=1),
                                np.expand_dims(x_a1, axis=1),
                                np.expand_dims(x_a2, axis=1)], axis=1)
    reg_params = np.linalg.lstsq(input_mtx, y, rcond=None)[0]
    yHat = reg_params[0] + reg_params[1] * x_a1 + reg_params[2] * x_a2
    SSE_full = ((y - yHat)**2).sum()
    f_score = (null_SSE - SSE_full) / (SSE_full / (len(x) - 3))
    return np.concatenate([[f_score], reg_params])
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _b_model(x, y, cp, null_SSE):

    x_b = np.concatenate([x[:cp], np.tile(x[cp], len(x) - cp)])
    input_mtx = np.concatenate([np.expand_dims(np.ones(len(x)), axis=1),
                                np.expand_dims(x_b, axis=1)], axis=1)
    reg_params = np.linalg.lstsq(input_mtx, y, rcond = None)[0]
    yHat = reg_params[0] + reg_params[1] * x_b
    full_SSE = ((y - yHat)**2).sum()
    f_score = (null_SSE - full_SSE) / (full_SSE / (len(x) - 2))
    return np.concatenate([[f_score], reg_params])
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _check_path_exists(path):

    try: assert os.path.isdir(path)
    except AssertionError: raise FileNotFoundError('Invalid path!')
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _cross_sample_stats_QC(df):

    """Calculate the cross-sample statistics for bootstrapped fit results"""

    years = df.Year.unique()
    stats_list = []
    trials_list = []
    for year in years:
        year_df = df.loc[df.Year == year]
        d_mode = len(df.loc[df.b1 > 0, 'b1'])
        e_mode = len(df.loc[df.b1 < 0, 'b1'])
        if e_mode > d_mode:
            df.loc[df.b1 > 0, ['ustar_th_b', 'b0', 'b1']] = np.nan
        else:
            df.loc[df.b1 < 0, ['ustar_th_b', 'b0', 'b1']] = np.nan
        norm_a1 = (
            (year_df.a1 * (year_df.ustar_th_a / (year_df.a0 + year_df.a1 *
                                                 year_df.ustar_th_a)))
            .median()
            )
        norm_a2 = (
            (year_df.a2 * (year_df.ustar_th_a / (year_df.a0 + year_df.a1 *
                                                 year_df.ustar_th_a)))
            .median())
        stats_list.append(
            {'Year': year, 'valid_n': year_df.ustar_th_b.count(),
             'norm_a1': norm_a1, 'norm_a2': norm_a2,
             'ustar_mean': year_df.ustar_th_b.mean(),
             'ustar_std': year_df.ustar_th_b.std()}
            )
        year_df = year_df[['bootstrap_n', 'b0', 'b1', 'ustar_th_b']].dropna()
        mean_df = year_df.groupby(['bootstrap_n']).mean()
        mean_df['ustar_std'] = (year_df.groupby(['bootstrap_n']).std()
                                ['ustar_th_b'])
        mean_df['valid_n'] = (year_df.groupby(['bootstrap_n']).count()
                              ['ustar_th_b'])
        mean_df['Year'] = year
        mean_df.rename({'ustar_th_b': 'ustar_mean'}, axis=1, inplace=True)
        mean_df.reset_index(inplace=True)
        mean_df = mean_df[['Year', 'bootstrap_n', 'valid_n', 'b0', 'b1',
                           'ustar_mean', 'ustar_std']]
        trials_list.append(mean_df)
    return {'trial_results': pd.concat(trials_list),
            'summary_statistics': pd.DataFrame(stats_list)}
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _define_default_external_names():

    return {'flux_name': 'Fc',
            'temperature_name': 'Ta',
            'insolation_name': 'Fsd',
            'friction_velocity_name': 'ustar'}
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _define_default_internal_names():

    return {'flux_name': 'NEE',
            'temperature_name': 'Ta',
            'insolation_name': 'Fsd',
            'friction_velocity_name': 'ustar'}
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def f_test(f_max, n, model):

    """Do f-test to see if model performance exceeds null model
       Args:
           * f_max (float): maximum f value found across all change points
           * n (int): size of sample
           * model (str): specify model used ('a' or 'b')"""

    p = np.nan
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
    p_bounds = [1 - (1 - x) / 2 for x in [cols[0], cols[-1]]]
    f_crit_vals = [float(PchipInterpolator(crit_table.index,
                                                        crit_table[x])(n)) for x in crit_table.columns]
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

#------------------------------------------------------------------------------
def fit_function(data_array, psig=0.05):

    # Configs and initialisations
    x_array, y_array = data_array[:, 0], data_array[:, 1]
    n_cases = len(data_array)
    iter_range = _get_iter_range(n_cases)
    results_dict = {}

    # Calculate null model SSE for operational (b) and diagnostic (a) model
    SSE_null_b = _null_b_model(y_array)
    SSE_null_a = _null_a_model(x_array, y_array)

    # Iterate through all possible change points for models; get max f-score,
    # associated change point and threshold value

    # A model
    results = np.array([_a_model(x_array, y_array, this_cp, SSE_null_a)
                        for this_cp in range(iter_range[0], iter_range[1])])
    fmax_a = results[:, 0].max()
    p_a = f_test(fmax_a, n_cases, model = 'a')
    if p_a < psig:
        cp_a = int(results[:, 0].argmax())
        a_dict = {'ustar_th_a': x_array[cp_a + iter_range[0]]}
        a_dict.update(dict(zip(['a0', 'a1', 'a2'], results[cp_a, 1:])))
    else:
        a_dict = {var: np.nan for var in ['ustar_th_a', 'a0', 'a1', 'a2']}

    # B model
    results = np.array([_b_model(x_array, y_array, this_cp, SSE_null_b)
                        for this_cp in range(iter_range[0], iter_range[1])])
    fmax_b = results[:, 0].max()
    p_b = f_test(fmax_b, n_cases, model = 'b')
    if p_b < psig:
        cp_b = int(results[:, 0].argmax())
        b_dict = {'ustar_th_b': x_array[cp_b + iter_range[0]]}
        b_dict.update(dict(zip(['b0', 'b1'], results[cp_b, 1:])))
    else:
        b_dict = {var: np.nan for var in ['ustar_th_b', 'b0', 'b1']}

    results_dict.update(a_dict)
    results_dict.update(b_dict)

    return results_dict
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _fit_season_data(seasons_df):

    iter_idx = (
        seasons_df.groupby(['Year', 'Season', 'Tclass']).mean().index
        )
    return (
        pd.DataFrame(
            [fit_function(seasons_df.loc[x, ['ustar', 'NEE']]
                          .to_numpy())
             for x in iter_idx],
            index = iter_idx)
        )
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _get_bin_index(season_stats):

    n_bins = season_stats['n_bins']
    n_per_bin = season_stats['n_per_bin']
    return np.tile(np.concatenate([np.tile(x, n_per_bin)
                                   for x in range(n_bins)]), 4)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# def get_ds_from_nc(nc_path):

#     ds = xr.open_dataset()

#     return
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _get_iter_range(n_cases):

    endpts_threshold = int(np.floor(n_cases * 0.05))
    if endpts_threshold < 3: endpts_threshold = 3
    return (endpts_threshold, n_cases - endpts_threshold)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def get_resampled_data(df):

    """Create new sample by randomly resampling original data"""

    if len(df) == 0: raise RuntimeError('No data available')
    random_index = sorted(np.random.randint(0, len(df) - 1, len(df)))
    return df.iloc[random_index]
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _get_season_data(df, stats_dict):

    # Extract overlapping series to individual dataframes, for each of
    # which: # 1) sort by temperature; 2) create temperature class;
    # 3) sort temperature class by u*; 4) add bin numbers to each class,
    # then; 5) concatenate
    df = df.iloc[:stats_dict['valid_n_incl']]
    season_index = _get_season_index(stats_dict)
    T_index = _get_Tclass_index(stats_dict)
    bin_index = _get_bin_index(stats_dict)
    seasons_lst = []
    for season in np.unique(season_index)[:-1]:
        this_df = (
            df.loc[(season_index >= season) &
                   (season_index < season + 2)]
            .sort_values('Ta', axis=0)
            .reset_index(drop=True)
            )
        this_df['Season'], this_df['Tclass'] = season, T_index
        this_df = (
            pd.concat([this_df.loc[this_df.Tclass == this_season]
                       .sort_values('ustar', axis = 0)
                       for this_season in range(4)])
            .reset_index(drop=True)
            )
        this_df['Bin'] = bin_index
        seasons_lst.append(this_df)
    seasons_df = pd.concat(seasons_lst)

    # Construct multiindex and use Season, T_class and Bin as levels,
    # drop them as df variables then average by bin and drop it from the
    # index; return the multi-indexed df (could drop T here if not useful?)
    name_list = ['Season', 'Tclass', 'Bin']
    multi_index = pd.MultiIndex.from_frame(seasons_df[name_list])
    seasons_df.drop(name_list, axis = 1, inplace = True)
    seasons_df.index = multi_index
    return (
        seasons_df.groupby(level = name_list).mean()
        .reset_index(level=['Bin'], drop=True)
        )
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _get_season_stats(df, freq, minimum_annual_n):

    """Get statistics of data availability for each year"""

    n_per_bin = 5 if freq == 30 else 3
    season_n = 1000 if freq == 30 else 600
    total_n = len(df)
    if total_n >= minimum_annual_n: sufficient_data = 'True'
    else: sufficient_data = 'False'
    standard_seasons = int(total_n / season_n)
    overlap_seasons = standard_seasons * 2 - 1
    remain = total_n % season_n
    min_n_per_season = n_per_bin * 4
    extra_n_per_season = remain / standard_seasons
    extra_n_per_season = int(extra_n_per_season - (extra_n_per_season %
                                                   min_n_per_season))
    n_per_season = season_n + extra_n_per_season
    n_per_Tclass = int(n_per_season / 4)
    n_bins = int(n_per_Tclass / n_per_bin)
    return (
        {'valid_data': 'True', 'sufficient_data': sufficient_data,
         'valid_n': total_n, 'valid_n_incl': standard_seasons * n_per_season,
         'overlap_seasons': overlap_seasons,
         'n_per_season': n_per_season, 'n_per_Tclass': n_per_Tclass,
         'n_bins': n_bins, 'n_per_bin': n_per_bin}
        )
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _get_season_index(season_stats):

    n_per_season = season_stats['n_per_season']
    n_seasons = season_stats['overlap_seasons']
    return np.concatenate([np.tile(i, int(n_per_season / 2))
                                   for i in range(n_seasons + 1)])
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _get_Tclass_index(season_stats):

    n_per_Tclass = season_stats['n_per_Tclass']
    return np.concatenate([np.tile(x, n_per_Tclass) for x in range(4)])
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _null_a_model(x, y):

    alpha0, alpha1 = np.polyfit(x, y, deg=1)
    return ((y - (x * alpha0 + alpha1))**2).sum()
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _null_b_model(y):

    return ((y - y.mean())**2).sum()
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _write_to_file(data_dict, write_dir):

    """Write the returned results to an excel file"""

    if not os.path.isdir(write_dir): os.mkdir(write_dir)
    path_file = os.path.join(write_dir, 'change_points.xlsx')
    xlwriter = pd.ExcelWriter(path_file)
    for key in sorted(data_dict.keys()):
        if key == 'trial_results' or 'raw_results':
            df = data_dict[key]
            for year in df.Year.unique():
                (df.loc[df.Year==year]
                 .drop('Year', axis=1)
                 .to_excel(xlwriter, sheet_name=str(year), index=None)
                 )
            continue
        data_dict[key].to_excel(xlwriter, sheet_name=key, index=None)
    xlwriter.save()
#------------------------------------------------------------------------------

# #--------------------------------------------------------------------------
# def plot_fit(df):

#     plot_df = df.copy().reset_index(drop = True)
#     stats_df = pd.DataFrame(fit(df), index = [0])
#     if stats_df.empty:
#         raise RuntimeError('Could not find a valid changepoint for this '
#                            'sample')
#     zero_list = [np.nan, 0, np.nan]
#     if 'ustar_th_b' in stats_df:
#         zero_list.append(stats_df.b0.item())
#         cp_b = np.where(df.ustar == stats_df.ustar_th_b.item())[0].item()
#         plot_df['yHat_b'] = (stats_df.ustar_th_b.item() * stats_df.b1.item() +
#                              stats_df.b0.item())
#         plot_df['yHat_b'].iloc[:cp_b] = (plot_df.ustar.iloc[:cp_b] *
#                                          stats_df.b1.item() +
#                                          stats_df.b0.item())

#     if 'ustar_th_a' in stats_df:
#         zero_list.append(stats_df.a0.item())
#         cp_a = np.where(df.ustar == stats_df.ustar_th_a.item())[0].item()
#         NEE_at_cp_a = (stats_df.ustar_th_a.item() * stats_df.a1.item() +
#                        stats_df.a0.item())
#         if 'ustar_th_a' in stats_df:
#             plot_df['yHat_a'] = (plot_df.ustar * stats_df.a1.item() +
#                                  stats_df.a0.item())
#             plot_df['yHat_a'].iloc[cp_a + 1:] = ((plot_df.ustar.iloc[cp_a + 1:] -
#                                                   stats_df.ustar_th_a.item()) *
#                                                  stats_df.a2.item() +
#                                                  NEE_at_cp_a)
#     plot_df.loc[-1] = zero_list
#     plot_df.index = plot_df.index + 1
#     plot_df = plot_df.sort_index()
#     fig, ax = plt.subplots(1, 1, figsize = (14, 8))
#     ax.set_xlim([0, plot_df.ustar.max() * 1.05])
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     ax.tick_params(axis = 'y', labelsize = 14)
#     ax.tick_params(axis = 'x', labelsize = 14)
#     fig.patch.set_facecolor('white')
#     ax.set_xlabel('$u*\/(m\/s^{-1}$)', fontsize = 16)
#     ax.set_ylabel('$NEE\/(\mu mol C\/m^{-2} s^{-1}$)', fontsize = 16)
#     ax.axhline(0, color = 'black', lw = 0.5)
#     ax.plot(plot_df.ustar, plot_df.NEE, 'bo', label = 'observational data')
#     if 'ustar_th_b' in stats_df:
#         ax.plot(plot_df.ustar, plot_df.yHat_b, color = 'red',
#                 label = 'operational model')
#     if 'ustar_th_a' in stats_df:
#         ax.plot(plot_df.ustar, plot_df.yHat_a, color = 'green',
#                 label = 'diagnostic model')
#     ax.legend(loc = (0.05, 0.85), fontsize = 12, frameon = False)
#     return
# #------------------------------------------------------------------------------