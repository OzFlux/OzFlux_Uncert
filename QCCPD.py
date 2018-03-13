# -*- coding: utf-8 -*-

# Python modules
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import datetime as dt
import numpy as np
import pandas as pd
from scipy import stats
import os
import sys
#from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator
import pdb

#------------------------------------------------------------------------------
# Class init
#------------------------------------------------------------------------------
class change_point_detect(object):
    
    def __init__(self, dataframe, n_bootstraps = 1):

        interval = int(filter(lambda x: x.isdigit(), 
                              pd.infer_freq(dataframe.index)))
        assert interval % 30 == 0
        self.df = dataframe
        self.interval = interval
        self.n_bootstraps = n_bootstraps

    #------------------------------------------------------------------------------
    # Coordinate steps in CPD process
    def main(self):
        """
        This script fetches data from an OzFluxQC .nc file and applies change point detection
        algorithms to the nocturnal C flux data to provide a best estimate for the u*threshold, 
        as well as associated uncertainties (95%CI). It stratifies the data by year, 'season'* 
        and temperature class (data are also binned to reduce noise) and the analysis runs 
        on each of the resulting samples. It is based on:
            
        Barr, A.G., Richardson, A.D., Hollinger, D.Y., Papale, D., Arain, M.A., Black, T.A., 
        Bohrer, G., Dragoni, D., Fischer, M.L., Gu, L., Law, B.E., Margolis, H.A., McCaughey, J.H., 
        Munger, J.W., Oechel, W., Schaeffer, K., 2013. Use of change-point detection for 
        friction–velocity threshold evaluation in eddy-covariance studies. 
        Agric. For. Meteorol. 171-172, 31–45. doi:10.1016/j.agrformet.2012.11.023
        
    
            
        * Season is just a 1000 point slice of nocturnal data - these slices also overlap by 50%.    
        """
       
        print 'Starting analysis...'
        
        final_results_df = pd.DataFrame()
    
        # Bootstrap the data and run the CPD algorithm
        for i in xrange(self.n_bootstraps):
                
            # First pass is obs, otherwise use bootstraps
            try:
                assert i > 0         
                work_df = pd.concat(map(lambda x: bootstrap(self.df.loc[str(x)]),
                                        list(set(self.df.index.year)))).copy()
                print 'Analysing bootstrap {}'.format(str(i))
            except:
                work_df = self.df.copy()
                print 'Analysing observational data for first pass'
            
            # Arrange data into seasons and analyze fits
            print 'Finding change points...'
            seasons_df = sort_data(work_df, self.interval)
            interim_results_df = pd.DataFrame({'Ta_mean': seasons_df['Ta'].
                                               groupby(['Year', 'Season', 
                                                        'T_class']).mean()})
            stats_df = pd.DataFrame(map(lambda x: fit(seasons_df.loc[x]), 
                                        interim_results_df.index),
                                    index = interim_results_df.index)
            interim_results_df = interim_results_df.join(stats_df)
            print 'Done!'
    
            interim_results_df = interim_results_df.reset_index(level=['Season',
                                                                       'T_class'], 
                                                                drop = True)
    
            final_results_df = pd.concat([final_results_df, interim_results_df])       
                
        print 'Finished change point detection for all bootstraps'
        print 'Starting QC'    
        
        final_results_df.sort_index(inplace = True)
        return final_results_df
#        
#        # Drop all years with no data remaining after QC, and return nothing if all years were dropped
#        [counts_df.drop(i,inplace=True) for i in counts_df.index if counts_df.loc[i, 'Total'] == 0]    
#        if counts_df.empty:
#            sys.exit('Insufficient data for analysis... exiting')
#    
#        # QC the combined results
#        print 'Doing cross-sample QC...'
#        output_stats_df = QC2(all_results_df, counts_df, d['num_bootstraps'])
#        print 'Done!' 
#    
#        # Calculate final values
#        print 'Calculating final results' 
#        output_stats_df = stats_calc(all_results_df, output_stats_df)
#        
#        # If requested by user, plot: 1) histograms of u* thresholds for each year; 
#        #                             2) normalised a1 and a2 values
#        if 'plot_output_path' in d.keys():
#            print 'Plotting u* histograms for all valid b model thresholds for all valid years'
#            [plot_hist(all_results_df.loc[j, 'bMod_threshold'][all_results_df.loc[j, 'b_valid'] == True],
#                       output_stats_df.loc[j, 'ustar_mean'],
#                       output_stats_df.loc[j, 'ustar_sig'],
#                       output_stats_df.loc[j, 'crit_t'],
#                       j, d['plot_output_path'])
#             for j in output_stats_df.index]
#            
#            print 'Plotting normalised median slope parameters for all valid a model thresholds for all valid years'
#            plot_slopes(output_stats_df[['norm_a1_median', 'norm_a2_median']], d['plot_output_path'])    
#        
#        # Output final stats if requested by user
#        if 'results_output_path' in d.keys():
#            print 'Outputting final results'
#            output_stats_df.to_csv(os.path.join(d['results_output_path'], 'annual_statistics.csv'))    
#        
#        print 'Analysis complete!'
#        # Return final results
#        return output_stats_df    
#    #------------------------------------------------------------------------------

    #------------------------------------------------------------------------------
def sort_data(df, interval):

    season_n = 1000 if interval == 30 else 600
    bin_n = 5 if interval == 30 else 3

    # Create a df containing count stats for the variables for all available years
    df = df.loc[df['Fsd'] < 10, ['Fc', 'ustar', 'Ta']].dropna().copy()
    years_df = df[['Fc', 'ustar']].groupby([lambda x: x.year]).count()
    years_df.drop('ustar', axis = 1, inplace = True)
    years_df.columns = ['n_valid']
    years_df['n_seasons'] = map(lambda x: years_df.loc[x, 'n_valid'] / 
                                (season_n / 2) - 1, years_df.index)
    years_df['n_seasons'].fillna(0, inplace=True)
    if np.all(years_df['n_seasons'] == 0):
        print('No years with sufficient data for evaluation. Returning...')
        return
    if np.any(years_df['n_seasons'] == 0):
        exclude_years_list = years_df[years_df['n_seasons'] <= 0].index.tolist()
        exclude_years_str= ','.join(map(str, exclude_years_list))
        print ('Insufficient data for evaluation in the following years: {}'
               ' (excluded from analysis)'.format(exclude_years_str))
        years_df = years_df.loc[years_df['seasons'] > 0]

    # Extract overlapping series to individual dataframes, for each of which: 
    # 1) sort by temperature; 2) create temperature class; 3) sort temperature 
    # class by u*; 4) add bin numbers to each class, then; 5) concatenate
    lst = []
    T_array = np.concatenate(map(lambda x: np.tile(x, season_n / 4), range(4)))
    bin_array = np.tile(np.concatenate(map(lambda x: np.tile(x, bin_n), 
                                           range(50))), 4)
    for year in years_df.index:
        for season in xrange(years_df.loc[year, 'n_seasons']):
            start_ind = season * (season_n / 2)
            end_ind = season * (season_n / 2) + season_n
            this_df = df.loc[str(year)].iloc[start_ind: end_ind]
            this_df.sort_values('Ta', axis = 0, inplace = True)
            this_df['Year'] = this_df.index.year
            this_df['Season'] = season + 1
            this_df['T_class'] = T_array
            this_df = pd.concat(map(lambda x: 
                                    this_df.loc[this_df.T_class == x]
                                    .sort_values('ustar', axis = 0), 
                                    range(4)))
            this_df['Bin'] = bin_array
            lst.append(this_df)
    seasons_df = pd.concat([frame for frame in lst])

    # Construct multiindex and use Year, Season, T_class and Bin as levels,
    # drop them as df variables then average by bin and drop it from the index
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

    return seasons_df
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Return a bootstrapped sample of the passed dataframe
def bootstrap(df):
    return df.iloc[sorted(np.random.randint(0, len(df) - 1, len(df)))]
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def critical_f(f_max, n, model):
    
    p = np.NaN
    assert ~np.isnan(f_max)
    assert ~np.isnan(n)
    assert n > 10
    
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
        
    elif model == 'a':
        
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

    crit_table = pd.DataFrame(arr, index = idx, columns = cols)
    p_bounds = map(lambda x: 1 - (1 - x) / 2, [cols[0], cols[-1]])
    f_crit_vals = map(lambda x: float(PchipInterpolator(crit_table.index, 
                                                        crit_table[x])(n)), 
                      crit_table.columns)
    if f_max < f_crit_vals[0]:
        f_adj = stats.f.ppf(p_bounds[0], len(cols), n) * f_max / f_crit_vals[0]
        p = 2 * (1 - stats.f.cdf(f_adj, len(cols), n))
        if p > 1: p = 1 
    elif f_max > f_crit_vals[-1]:
        f_adj = stats.f.ppf(p_bounds[-1], len(cols), n) * f_max / f_crit_vals[-1]
        p = 2 * (1 - stats.f.cdf(f_adj, len(cols), n))
        if p < 0: p = 0
    else:
        p = PchipInterpolator(f_crit_vals, (1 - np.array(cols)).tolist())(f_max)
    return p
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def fit(df):
    
    def a_model_statistics(cp):
        work_df = temp_df.copy()
        work_df['ustar_a1'] = work_df['ustar']
        work_df['ustar_a1'].iloc[cp + 1:] = work_df['ustar_a1'].iloc[cp]
        dummy_array = np.concatenate([np.zeros(cp + 1), 
                                      np.ones(df_length - (cp + 1))])
        work_df['ustar_a2'] = (work_df['ustar'] - 
                               work_df['ustar'].iloc[cp]) * dummy_array
        reg_params = np.linalg.lstsq(work_df[['int','ustar_a1','ustar_a2']],
                                     work_df['Fc'])[0]
        yHat = (reg_params[0] + reg_params[1] * work_df['ustar_a1'] +
                reg_params[2] * work_df['ustar_a2'])
        SSE_full = ((work_df['Fc'] - yHat)**2).sum()
        f_score = (SSE_null_a - SSE_full) / (SSE_full / (df_length - 2))
        return f_score, reg_params
    
    def b_model_statistics(cp):
        work_df = temp_df.copy()
        work_df['ustar_b'] = work_df['ustar']
        work_df['ustar_b'].iloc[cp + 1:] = work_df['ustar_b'].iloc[cp]
        reg_params = np.linalg.lstsq(work_df[['int','ustar_b']], 
                                     work_df['Fc'])[0]
        yHat = reg_params[0] + reg_params[1] * work_df['ustar_b']
        SSE_full = ((work_df['Fc'] - yHat)**2).sum()
        f_score = (SSE_null_b - SSE_full) / (SSE_full / (df_length - 2))
        return f_score, reg_params
        
    # Get stuff ready
    temp_df = df.copy()
    temp_df = temp_df.reset_index(drop = True)
    temp_df = temp_df.astype(np.float64)        
    df_length = len(temp_df)
    endpts_threshold = np.floor(df_length * 0.05)
    if endpts_threshold < 3: endpts_threshold = 3
    psig = 0.05
    
    # Calculate null model SSE for operational (b) and diagnostic (a) model
    SSE_null_b = ((temp_df['Fc'] - temp_df['Fc'].mean())**2).sum()
    alpha0 , alpha1 = stats.linregress(temp_df['ustar'], temp_df['Fc'])[:2]
    SSE_null_a = ((temp_df['Fc'] - (temp_df['ustar'] * 
                                    alpha0 + alpha1))**2).sum()
    
    # Create arrays to hold statistics
    f_a_array = np.zeros(df_length)
    f_b_array = np.zeros(df_length)
    
    # Add series to df for numpy linalg
    temp_df['int'] = np.ones(df_length)
        
    # Iterate through all possible change points 
    for i in xrange(1, df_length - 1):
              
        # Diagnostic (a) and operational (b) model statistics
        f_a_array[i] = a_model_statistics(i)[0]
        f_b_array[i] = b_model_statistics(i)[0]

    # Get max f-score, associated change point and ustar value for models
    # (conditional on passing f score and end points within limits)
    d = {}
    fmax_a, cp_a = f_a_array.max(), int(f_a_array.argmax())
    if ((cp_a > endpts_threshold) | (cp_a < df_length - endpts_threshold)):
        p_a = critical_f(fmax_a, df_length, model = 'a')
        if p_a < psig:
            d['ustar_th_a'] = temp_df['ustar'].iloc[cp_a]
            d['a0'], d['a1'], d['a2'] = a_model_statistics(cp_a)[1] 

#            d['norm_a1'] = a1 * (ustar_th_a / (a0 + a1 * ustar_th_a))
#            d['norm_a2'] = a2 * (ustar_th_a / (a0 + a1 * ustar_th_a))
                      
    fmax_b, cp_b = f_b_array.max(), int(f_b_array.argmax())
    if ((cp_b > endpts_threshold) | (cp_b < df_length - endpts_threshold)):
        p_b = critical_f(fmax_b, len(temp_df), model = 'b')
        if p_b < psig:    
            d['ustar_th_b'] = temp_df['ustar'].iloc[cp_b]
            d['b0'], d['b1'] = b_model_statistics(cp_b)[1]

    return d

#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Plot identified change points in observed (i.e. not bootstrapped) data and   
# write to specified folder                                                    
def plot_fits(temp_df,stats_df,plot_out):
    
    # Create series for use in plotting (this could be more easily called from fitting function - why are we separating these?)
    temp_df['ustar_alt']=temp_df['ustar']
    temp_df['ustar_alt'].iloc[int(stats_df['bMod_CP'])+1:]=stats_df['bMod_threshold']
    temp_df['ustar_alt1']=temp_df['ustar']
    temp_df['ustar_alt1'].iloc[stats_df['aMod_CP']+1:]=temp_df['ustar_alt1'].iloc[stats_df['aMod_CP']]
    temp_df['ustar_alt2']=((temp_df['ustar']-stats_df['aMod_threshold'])
                           *np.concatenate([np.zeros(stats_df['aMod_CP']+1),np.ones(50-(stats_df['aMod_CP']+1))]))
    temp_df['yHat_a']=stats_df['a0']+stats_df['a1']*temp_df['ustar_alt1']+stats_df['a2']*temp_df['ustar_alt2'] # Calculate the estimated time series
    temp_df['yHat_b']=stats_df['b0']+stats_df['b1']*temp_df['ustar_alt']          
    
    # Now plot    
    fig=plt.figure(figsize=(12,8))
    fig.patch.set_facecolor('white')
    plt.plot(temp_df['ustar'],temp_df['Fc'],'bo')
    plt.plot(temp_df['ustar'],temp_df['yHat_b'],color='red')   
    plt.plot(temp_df['ustar'],temp_df['yHat_a'],color='green')   
    plt.title('Year: '+str(stats_df.name[0])+', Season: '+str(stats_df.name[1])+', T class: '+str(stats_df.name[2])+'\n',fontsize=22)
    plt.xlabel(r'u* ($m\/s^{-1}$)',fontsize=16)
    plt.ylabel(r'Fc ($\mu mol C\/m^{-2} s^{-1}$)',fontsize=16)
    plt.axvline(x=stats_df['bMod_threshold'],color='black',linestyle='--')
    props = dict(boxstyle='round,pad=1', facecolor='white', alpha=0.5)
    txt='Change point detected at u*='+str(round(stats_df['bMod_threshold'],3))+' (i='+str(stats_df['bMod_CP'])+')'
    ax=plt.gca()
    plt.text(0.57,0.1,txt,bbox=props,fontsize=12,verticalalignment='top',transform=ax.transAxes)
    plot_out_name='Y'+str(stats_df.name[0])+'_S'+str(stats_df.name[1])+'_Tclass'+str(stats_df.name[2])+'.jpg'
    fig.savefig(os.path.join(plot_out,plot_out_name))
    plt.close(fig)

# Plot PDF of u* values and write to specified folder           
def plot_hist(S,mu,sig,crit_t,year,plot_out):
    S=S.reset_index(drop=True)
    x_low=S.min()-0.1*S.min()
    x_high=S.max()+0.1*S.max()
    x=np.linspace(x_low,x_high,100)
    fig=plt.figure(figsize=(12,8))
    fig.patch.set_facecolor('white')
    plt.hist(S,normed=True)
    plt.plot(x,mlab.normpdf(x,mu,sig),color='red',linewidth=2.5,label='Gaussian PDF')
    plt.xlim(x_low,x_high)
    plt.xlabel(r'u* ($m\/s^{-1}$)',fontsize=16)
    plt.axvline(x=mu-sig*crit_t,color='black',linestyle='--')
    plt.axvline(x=mu+sig*crit_t,color='black',linestyle='--')
    plt.axvline(x=mu,color='black',linestyle='dotted')
    props = dict(boxstyle='round,pad=1', facecolor='white', alpha=0.5)
    txt='mean u*='+str(mu)
    ax=plt.gca()
    plt.text(0.4,0.1,txt,bbox=props,fontsize=12,verticalalignment='top',transform=ax.transAxes)
    plt.legend(loc='upper left')
    plt.title(str(year)+'\n')
    plot_out_name='ustar'+str(year)+'.jpg'
    fig.savefig(os.path.join(plot_out,plot_out_name))
    plt.close(fig)

# Plot normalised slope parameters to identify outlying years and output to    
# results folder - user can discard output for that year                       
def plot_slopes(df,plot_out):
    df=df.reset_index(drop=True)
    fig=plt.figure(figsize=(12,8))
    fig.patch.set_facecolor('white')
    plt.scatter(df['norm_a1_median'],df['norm_a2_median'],s=80,edgecolors='blue',facecolors='none')
    plt.xlim(-4,4)
    plt.ylim(-4,4)
    plt.xlabel('$Median\/normalised\/ a^{1}$',fontsize=16)
    plt.ylabel('$Median\/normalised\/ a^{2}$',fontsize=16)
    plt.title('Normalised slope parameters \n')
    plt.axvline(x=1,color='black',linestyle='dotted')
    plt.axhline(y=0,color='black',linestyle='dotted')
    plot_out_name='normalised_slope_parameters.jpg'
    fig.savefig(os.path.join(plot_out,plot_out_name))
    plt.close(fig)

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Quality control within bootstrap
def QC1(QC1_df):
       
    QC1_df['major_mode'] = True

    # For each year, find all cases that belong to minority mode (i.e. mode is sign of slope below change point)
    total_count = QC1_df['bMod_threshold'].groupby(level = 'year').count()
    neg_slope = QC1_df['bMod_threshold'][QC1_df['b1'] < 0].groupby(level = 'year').count()
    neg_slope = neg_slope.reindex(total_count.index)
    neg_slope = neg_slope.fillna(0)
    neg_slope = neg_slope/total_count * 100
    for i in neg_slope.index:
        sign = 1 if neg_slope.loc[i] < 50 else -1
        QC1_df.loc[i, 'major_mode'] = np.sign(np.array(QC1_df.loc[i, 'b1'])) == sign
    


    # Return the results df
    QC1_df = QC1_df.drop('major_mode', axis = 1)
    return QC1_df
    
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Quality control across bootstraps
def QC2(df,output_df,bootstrap_n):
    
    # Get the median values of the normalised slope parameters for each year
    output_df['norm_a1_median']=df['norm_a1'][df['a_valid']==True].groupby(df[df['a_valid']==True].index).median()
    output_df['norm_a2_median']=df['norm_a2'][df['a_valid']==True].groupby(df[df['a_valid']==True].index).median()
    
    # Get the proportion of all available cases that passed QC for b model   
    output_df['QCpass']=df['bMod_threshold'][df['b_valid']==True].groupby(df[df['b_valid']==True].index).count()
    output_df['QCpass_prop']=output_df['QCpass']/output_df['Total']
    
    # Identify years where either diagnostic or operational model did not find enough good data for robust estimate
    output_df['a_valid']=(~(np.isnan(output_df['norm_a1_median']))&(~np.isnan(output_df['norm_a2_median'])))
    output_df['b_valid']=(output_df['QCpass']>(4*bootstrap_n))&(output_df['QCpass_prop']>0.2)
    for i in output_df.index:
        if output_df['a_valid'].loc[i]==False: 
            print 'Insufficient valid cases for robust diagnostic (a model) u* determination in year '+str(i)
        if output_df['b_valid'].loc[i]==False: 
            print 'Insufficient valid cases for robust operational (b model) u* determination in year '+str(i)
 
    return output_df    
    
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
def stats_calc(df,stats_df):
    
    # Add statistics vars to output df
    stats_df['ustar_mean'] = np.nan
    stats_df['ustar_sig'] = np.nan
    stats_df['ustar_n'] = np.nan
    stats_df['crit_t'] = np.nan
    stats_df['95%CI_lower'] = np.nan
    stats_df['95%CI_upper'] = np.nan
    stats_df['skew'] = np.nan
    stats_df['kurt'] = np.nan
        
    # Drop data that failed b model, then drop b model boolean variable
    df=df[df['b_valid']==True]
    df=df.drop('b_valid',axis=1)
 
    # Calculate stats
    for i in stats_df.index:
        if stats_df.loc[i, 'b_valid']:
            if isinstance(df.loc[i, 'bMod_threshold'],pd.Series):
                temp = stats.describe(df.loc[i, 'bMod_threshold'])
                stats_df.loc[i, 'ustar_mean'] = temp[2]
                stats_df.loc[i, 'ustar_sig'] = np.sqrt(temp[3])
                stats_df.loc[i, 'crit_t'] = stats.t.ppf(1 - 0.025, temp[0])
                stats_df.loc[i, '95%CI_lower'] = (stats_df.loc[i, 'ustar_mean'] - 
                                                  stats_df.loc[i, 'ustar_sig'] * 
                                                  stats_df.loc[i, 'crit_t'])
                stats_df.loc[i, '95%CI_upper'] = (stats_df.loc[i, 'ustar_mean'] + 
                                                  stats_df.loc[i, 'ustar_sig'] *
                                                  stats_df.loc[i, 'crit_t'])
                stats_df.loc[i, 'skew'] = temp[4]
                stats_df.loc[i, 'kurt'] = temp[5]
            else:
                stats_df.loc[i, 'ustar_mean'] = df.loc[i, 'bMod_threshold']
    stats_df.index.name = 'Year'
                
    return stats_df
#------------------------------------------------------------------------------
    
if __name__=='__main__':
    test = main()