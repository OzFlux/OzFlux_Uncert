# -*- coding: utf-8 -*-
import os
import copy
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pdb

# Custom module imports
import random_plots

###### References ######

# Hollinger, D.Y., Richardson, A.D., 2005. Uncertainty in eddy covariance measurements and its application to physiological models. Tree Physiol. 25, 873–885.

# What?
def calculate_random_error(df,results_path_out,plot_path_out,configs,flux_frequency):
	
    print '\n------------------------'
    print 'Calculating random error'
    print '------------------------\n'
	
    # Calculate number of records in a day
    records_per_day=1440/int(flux_frequency)
	
    # Unpack the constraints etc
    Ta_diff=int(configs['temperature_difference_threshold'])
    ws_diff=int(configs['windspeed_difference_threshold'])
    rad_diff=int(configs['radiation_difference_threshold'])
    num_classes=int(configs['averaging_bins'])
	
    # Do the paired difference analysis (drop all cases containing any NaN - note that must convert shifted data to 
    # array to prevent error in hist plot - is this a bug?)
    diff_df=pd.DataFrame({'Fc_mean':(df['Fc']+df['Fc'].shift(records_per_day))/2,
 			  'Fc_diff':np.array(df['Fc']-df['Fc'].shift(records_per_day)),
			  'Ta':abs(df['Ta']-df['Ta'].shift(records_per_day)),
			  'ws':abs(df['ws']-df['ws'].shift(records_per_day)),
			  'Fsd':abs(df['Fsd']-df['Fsd'].shift(records_per_day))}).reset_index()
    diff_df.dropna(axis=0,how='any',inplace=True)
	
    # Find number of passed tuples, report, then drop everything except mean and difference of Fc tuples
    diff_df['pass_constraints']=(diff_df['Ta']<Ta_diff)&(diff_df['ws']<ws_diff)&(diff_df['Fsd']<rad_diff)
    total_tuples=len(diff_df)
    passed_tuples=len(diff_df[diff_df['pass_constraints']])	
    print (str(passed_tuples)+' of '+str(total_tuples)+' available tuples passed difference constraints (Fsd='+
  	    str(rad_diff)+'Wm^-2, Ta='+str(Ta_diff)+'C, ws='+str(ws_diff)+'ms^-1)\n')
    diff_df=diff_df[['Fc_mean','Fc_diff']][diff_df['pass_constraints']]
		
    # Cut into desired number of quantiles and get the flux and flux error means for each category
    diff_df['Category']=pd.qcut(diff_df['Fc_mean'],num_classes).labels
    diff_df.index=diff_df['Category']
    cats_df=pd.DataFrame({'Fc_mean':diff_df['Fc_mean'].groupby(diff_df['Category']).mean()})
    cats_df['sig_del']=np.nan
    for i in cats_df.index:
   	cats_df['sig_del'].ix[i]=(abs(diff_df['Fc_diff'].ix[i]-diff_df['Fc_diff'].ix[i].mean())).mean()*np.sqrt(2)
		
    # Calculate linear fit for +ve and -ve values...
    linreg_stats_df=pd.DataFrame(columns=['slope','intcpt','r_val','p_val','SE'],index=['influx','efflux'])
    linreg_stats_df.ix['efflux']=stats.linregress(cats_df['Fc_mean'][cats_df['Fc_mean']>0],cats_df['sig_del'][cats_df['Fc_mean']>0])
    linreg_stats_df.ix['influx']=stats.linregress(cats_df['Fc_mean'][cats_df['Fc_mean']<0],cats_df['sig_del'][cats_df['Fc_mean']<0])
    
    # Output results
    linreg_stats_df.to_csv(os.path.join(results_path_out,'random_error_stats.csv'))
    
    # Output plots	
    fig=random_plots.error_main(diff_df['Fc_diff'],cats_df,linreg_stats_df,num_classes)
    fig.savefig(os.path.join(plot_path_out,'Random_error_plots.jpg'))

    # Return data
    return linreg_stats_df

# What?
def propagate_random_error(prop_df,stats_df,results_path_out,plot_path_out,flux_frequency,num_trials):
     						 						 						
    # Create df for propagation of error (drop nans)
    prop_df['sig_del']=np.where(prop_df['Fc']>0,
                               (prop_df['Fc']*stats_df.ix['efflux'][0]+stats_df.ix['efflux'][1])/np.sqrt(2),
			       (prop_df['Fc']*stats_df.ix['influx'][0]+stats_df.ix['influx'][1])/np.sqrt(2))
    prop_df.dropna(inplace=True)
      
    # Calculate critical t
    crit_t=stats.t.isf(0.025,num_trials) #Calculate critical t-value for p=0.095
                
    # Calculate uncertainty due to random error for all measured data at annual time step
    years_df=pd.DataFrame({'Valid_n':prop_df['Fc'].groupby([lambda x: x.year]).count()})
    years_df['Random error (Fc)']=np.nan
    for i in years_df.index:
        temp_arr=np.array([np.random.laplace(0,prop_df['sig_del'].ix[str(i)]).sum() for j in xrange(num_trials)])
        years_df['Random error (Fc)'].ix[i]=temp_arr.std()*crit_t*12*10**-6*flux_frequency*60
    
    # Output data
    years_df.to_csv(os.path.join(results_path_out,'random_error_propagation_results.csv'))
    
    #Output plots
    
    # Return data
    return prop_df