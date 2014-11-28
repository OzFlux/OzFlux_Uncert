# -*- coding: utf-8 -*-
import os
import copy
import math
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pdb

#-----------------------------------------------------------------------------#

def calculate_random_error(df,configs):
#"""
## Pass the following arguments: 1) df containing non-gap filled QC'd Fc, Fsd, Ta, ws
#                                2) config file (dictionary) containing options as 
#                                   specified below
#
## Returns the linear regression statistics for daytime and nocturnal data
#
## Dictionary should contain the following:
#    'results_out_path' (path for results to be written to)
#    'flux_freq' (frequency of flux data)
#    'averaging_bins' (for calculation of random error variance as function of flux magnitude)    
#    'radiation_difference_threshold' (the maximum allowable Fsd difference between Fc pairs;
#                                      see below for threshold values)
#    'temperature_difference_threshold' (as above)
#    'windspeed_difference_threshold' (as above)
#                                 
## Algorithm from reference:
#    Hollinger, D.Y., Richardson, A.D., 2005. Uncertainty in eddy covariance measurements 
#    and its application to physiological models. Tree Physiol. 25, 873–885.
#
## Uses daily differencing procedure to estimate random error (note this will overestimate by factor of up to 2)
#
## Fc pairs must pass difference constraints as follows (as suggested in ref above):
#    Fsd:35W m^-2
#    Ta: 3C
#    ws: 1m s^-1
#"""	
    print '\n------------------------'
    print 'Calculating random error'
    print '------------------------\n'
	
    # Unpack the constraints etc
    Ta_threshold=int(configs['options']['temperature_difference_threshold'])
    ws_threshold=int(configs['options']['windspeed_difference_threshold'])
    rad_threshold=int(configs['options']['radiation_difference_threshold'])
    num_classes=int(configs['options']['averaging_bins'])
    num_trials=int(configs['options']['num_trials'])
    propagate_error=configs['options']['propagate_error']=='True'
    noct_threshold=int(configs['options']['noct_threshold'])
    records_per_day=1440/int(configs['options']['flux_frequency'])
    results_out_path=configs['files']['results_out_path']
    Fc=configs['variables']['Fc']
    Fsd=configs['variables']['Fsd']    
    Ta=configs['variables']['Ta']
    ws=configs['variables']['ws']
    
    # Do the paired difference analysis (drop all cases containing any NaN)
    diff_df=pd.DataFrame({'Fc_mean':(df[Fc]+df[Fc].shift(records_per_day))/2,
                          'Fc_diff':df[Fc]-df[Fc].shift(records_per_day),
        			  'Ta_diff':abs(df[Ta]-df[Ta].shift(records_per_day)),
        			  'ws_diff':abs(df[ws]-df[ws].shift(records_per_day)),
        			  'Fsd_diff':abs(df[Fsd]-df[Fsd].shift(records_per_day)),
                          'Day_ind':(df[Fsd]+df[Fsd].shift(records_per_day))/2 > noct_threshold}).reset_index()
    diff_df.dropna(axis=0,how='any',inplace=True)
    
    # Find number of passed tuples, report, then drop everything except mean and difference of Fc tuples
    diff_df['pass_constraints']=(diff_df['Ta_diff']<Ta_threshold)&(diff_df['ws_diff']<ws_threshold)&(diff_df['Fsd_diff']<rad_threshold)
    total_tuples=len(diff_df)
    passed_tuples=len(diff_df[diff_df['pass_constraints']])	
    print (str(passed_tuples)+' of '+str(total_tuples)+' available tuples passed difference constraints (Fsd='+
  	     str(rad_threshold)+'Wm^-2, Ta='+str(Ta_threshold)+'C, ws='+str(ws_threshold)+'ms^-1)\n')
    diff_df=diff_df[['Fc_mean','Fc_diff','Day_ind']][diff_df['pass_constraints']]
    diff_df['Class']=np.nan    
    
    # Calculate and report n for each bin
    n_per_class=int(len(diff_df)/num_classes)
    print 'Number of observations per bin = '+str(n_per_class)
    
    # Calculate and report nocturnal and daytime share of data
    day_classes=int(len(diff_df[diff_df['Day_ind']])/float(len(diff_df))*num_classes)
    noct_classes=num_classes-day_classes
    print 'Total bins = '+str(num_classes)+'; day bins = '+str(day_classes)+'; nocturnal bins = '+str(noct_classes)
    
    # Cut into desired number of quantiles and get the flux and flux error means for each category
    diff_df['Class'][diff_df['Day_ind']]=pd.qcut(diff_df['Fc_mean'][diff_df['Day_ind']],day_classes).labels
    diff_df['Class'][~diff_df['Day_ind']]=pd.qcut(diff_df['Fc_mean'][~diff_df['Day_ind']],noct_classes).labels+day_classes
    diff_df.index=diff_df['Class'].astype('int64')
    cats_df=pd.DataFrame({'Fc_mean':diff_df['Fc_mean'].groupby(diff_df['Class']).mean()})
    cats_df['sig_del']=np.nan
    cats_df['Day_ind']=np.nan
    cats_df['Day_ind'].ix[:day_classes]=True
    cats_df['Day_ind'].ix[day_classes:]=False
    for i in cats_df.index:
        i=int(i)
        cats_df['sig_del'].ix[i]=(abs(diff_df['Fc_diff'].ix[i]-diff_df['Fc_diff'].ix[i].mean())).mean()*np.sqrt(2)
    
    # Remove daytime positive bin averages and nocturnal negative bin averages
    cats_df['exclude']=((cats_df.Fc_mean>0)&(cats_df.Day_ind))|((cats_df.Fc_mean<0)&(cats_df.Day_ind==False))
    cats_df=cats_df[['Fc_mean','sig_del','Day_ind']][~cats_df.exclude]
    
    # Calculate linear fit for day and night values...
    linreg_stats_df=pd.DataFrame(columns=['slope','intcpt','r_val','p_val','SE'],index=['day','noct'])
    linreg_stats_df.ix['day']=stats.linregress(cats_df['Fc_mean'][cats_df['Day_ind']],cats_df['sig_del'][cats_df['Day_ind']])
    linreg_stats_df.ix['noct']=stats.linregress(cats_df['Fc_mean'][cats_df['Day_ind']==False],cats_df['sig_del'][cats_df['Day_ind']==False])
    
    print '\nRegression results (also saved at '+results_out_path+'):'
    print linreg_stats_df    
    
    # Output results
    linreg_stats_df.to_csv(os.path.join(results_out_path,'random_error_stats.csv'))
    
    # Output plots
    print '\nPlotting: 1) PDF of random error'
    print '          2) sigma_delta (variance of random error) as a function of flux magnitude'
    fig=main_plot(diff_df['Fc_diff'],cats_df,linreg_stats_df,num_classes)
    fig.savefig(os.path.join(results_out_path,'Random_error_plots.jpg'))

    # Return data
    if propagate_error:
        uncert_df=propagate_random_error(df,linreg_stats_df,configs)        
        return linreg_stats_df,uncert_df    
    else:
        return linreg_stats_df

# What?
def propagate_random_error(df,stats_df,configs):
    
    # Unpack the constraints etc
    num_trials=int(configs['options']['num_trials'])
    results_out_path=configs['files']['results_out_path']
    Fc=configs['variables']['Fc']
    flux_frequency=int(configs['options']['flux_frequency'])
    
    # Create df for propagation of error (drop nans)
    prop_df=pd.DataFrame({'Fc':df[Fc]})
    prop_df['sig_del']=np.where(prop_df[Fc]>=0,
                               (prop_df[Fc]*stats_df.ix['noct'][0]+stats_df.ix['noct'][1])/np.sqrt(2),
			             (prop_df[Fc]*stats_df.ix['day'][0]+stats_df.ix['day'][1])/np.sqrt(2))
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
    print '\nAnnual uncertainty in gC m-2 (also saved at '+results_out_path+'):'
    print years_df
    years_df.to_csv(os.path.join(results_out_path,'annual_uncertainty.csv'))
    prop_df['sig_del'].to_csv(os.path.join(results_out_path,'timestep_uncertainty.csv'))
   
    # Return data
    return prop_df
#-----------------------------------------------------------------------------#

# Plotting
#-----------------------------------------------------------------------------#
def myround(x,base=10):
    return int(base*round(x/base))

# Histogram to check error distribution is approximately Laplacian
def hist_plot(Fc_diff,ax):
	
    # Calculate scaling parameter for Laplace distribution over entire dataset (sigma / sqrt(2))
    beta=(abs(Fc_diff-Fc_diff.mean())).mean()
    
    # Calculate scaling parameter for Gaussian distribution over entire dataset (sigma)
    sig=Fc_diff.std()

    # Get edge quantiles and range, then calculate Laplace pdf over range
    x_low=myround(Fc_diff.quantile(0.005))
    x_high=myround(Fc_diff.quantile(0.995))
    x_range=(x_high-x_low)
    x=np.arange(x_low,x_high,1/(x_range*10.))
    pdf_laplace=np.exp(-abs(x/beta))/(2.*beta)
	    	
    # Plot normalised histogram with Laplacian and Gaussian pdfs
    ax.hist(np.array(Fc_diff),bins=300,range=[x_low,x_high],normed=True,color='blue',edgecolor='none')
    ax.plot(x,pdf_laplace,color='red',linewidth=2.5,label='Laplacian PDF')
    ax.plot(x,mlab.normpdf(x,0,sig),color='green',linewidth=2.5,label='Gaussian PDF')
    ax.set_xlabel(r'$\delta\/(\mu mol\/m^{-2} s^{-1}$)',fontsize=16)
    ax.legend(loc='upper left')
    ax.set_title('Random error distribution \n')

# Linear regression plot - random error variance as function of flux magnitude (binned by user-set quantiles)
def linear_plot(cats_df,linreg_stats_df,num_classes,ax):
    
    # Create series for regression lines
    influx_x=cats_df['Fc_mean'][cats_df['Fc_mean']<=0]
    influx_y=np.polyval([linreg_stats_df.ix['day'][0],linreg_stats_df.ix['day'][1]],influx_x)
    efflux_x=cats_df['Fc_mean'][cats_df['Fc_mean']>=0]
    efflux_y=np.polyval([linreg_stats_df.ix['noct'][0],linreg_stats_df.ix['noct'][1]],efflux_x)
    
    # Do plotting
    ax.plot(cats_df['Fc_mean'],cats_df['sig_del'],'ro')
    ax.plot(influx_x,influx_y,color='blue')
    ax.plot(efflux_x,efflux_y,color='blue')
    ax.set_xlim(round(cats_df['Fc_mean'].iloc[0])-1,math.ceil(cats_df['Fc_mean'].iloc[-1])+1)
    ax.set_xlabel(r'$C\/flux\/(\mu mol\/m^{-2} s^{-1}$)',fontsize=16)
    ax.set_title('Random error SD binned over flux magnitude quantiles (n='+str(num_classes)+')\n')
    
    # Move axis and relabel
    str_influx=('a='+str(round(linreg_stats_df.ix['day'][0],2))+
                '\nb='+str(round(linreg_stats_df.ix['day'][1],2))+
                '\nr$^2$='+str(round(linreg_stats_df.ix['day'][2],2))+
                '\np='+str(round(linreg_stats_df.ix['day'][3],2))+
                '\nSE='+str(round(linreg_stats_df.ix['day'][4],2)))
    str_efflux=('a='+str(round(linreg_stats_df.ix['noct'][0],2))+
                '\nb='+str(round(linreg_stats_df.ix['noct'][1],2))+
                '\nr$^2$='+str(round(linreg_stats_df.ix['noct'][2],2))+
                '\np='+str(round(linreg_stats_df.ix['noct'][3],2))+
                '\nSE='+str(round(linreg_stats_df.ix['noct'][4],2)))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.05, 0.35, str_influx, transform=ax.transAxes, fontsize=12,
            verticalalignment='top',bbox=props)
    ax.text(0.85, 0.35, str_efflux, transform=ax.transAxes, fontsize=12,
            verticalalignment='top',bbox=props)        
    ax.set_ylabel('$\sigma(\delta)$',fontsize=18)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticklabels([])
    ax2=ax.twinx()
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_position('zero')
    ax2.set_ylim(0,4.5)
	
# Create two plot axes
def main_plot(Fc_diff,cats_df,linreg_stats_df,num_classes):

    fig, (ax1, ax2) = plt.subplots(nrows=2)
    fig.patch.set_facecolor('white')
    hist_plot(Fc_diff, ax1)
    linear_plot(cats_df,linreg_stats_df,num_classes,ax2)
    fig.tight_layout()
    return fig
#-----------------------------------------------------------------------------#