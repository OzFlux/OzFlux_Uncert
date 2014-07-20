# -*- coding: utf-8 -*-
import os
import math
import copy
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

###### References ######

# Hollinger, D.Y., Richardson, A.D., 2005. Uncertainty in eddy covariance measurements and its application to physiological models. Tree Physiol. 25, 873–885.



###### Functions ######

# Calculate tuple differences
def paired_differences():
    diff_df=pd.DataFrame({'flux_err':np.nan,'flux_mean':np.nan},index=df.index)[:len(df)-recs_day]
    # for i in xrange(0,len(df)-recs_day):
        # rad_diff=abs(df[radName].iloc[i]-df[radName].iloc[i+recs_day])
        # temp_diff=abs(df[tempName].iloc[i]-df[tempName].iloc[i+recs_day])
        # wind_diff=abs(df[windName].iloc[i]-df[windName].iloc[i+recs_day])
        # if rad_diff<rad_threshold and temp_diff<temp_threshold and wind_diff<wind_threshold:
            # diff_df['flux_err'][i]=(df[CfluxName][i]-df[CfluxName][i+recs_day])/np.sqrt(2)
            # diff_df['flux_mean'][i]=(df[CfluxName][i]+df[CfluxName][i+recs_day])/2
    # diff_df=diff_df.dropna(how='any',axis=0)
    # hist_S=copy.copy(diff_df['flux_err'])
    # diff_df['flux_err']=abs(diff_df['flux_err'])
    # return diff_df,hist_S

def random_sample_laplace(df):
    crit_t=stats.t.isf(0.025,len(df)) #Calculate critical t-value for p=0.095
    return np.array([np.random.laplace(0,df).sum() for i in xrange(n)]).std()*crit_t

def hist_plot():
    
    plt.subplot(211)
    
    # Calculate scaling parameter for Laplace distribution over entire dataset (sigma / sqrt(2))
    beta=(abs(hist_S-hist_S.mean())).mean()
    
    # Calculate scaling parameter for Gaussian distribution over entire dataset (sigma)
    sig=hist_S.std()

    # Get edge quantiles and range, then calculate Laplace pdf over range
    x_low=myround(hist_S.quantile(0.005))
    x_high=myround(hist_S.quantile(0.995))
    x_range=(x_high-x_low)
    x=np.arange(x_low,x_high,1/(x_range*10.))
    pdf_laplace=np.exp(-abs(x/beta))/(2.*beta)
    
    # Plot normalised histogram with Laplacian and Gaussian pdfs
    plt.hist(hist_S,bins=300,range=[x_low,x_high],normed=True,color='blue',edgecolor='none')
    plt.plot(x,pdf_laplace,color='red',linewidth=2.5,label='Laplacian PDF')
    plt.plot(x,mlab.normpdf(x,0,sig),color='green',linewidth=2.5,label='Gaussian PDF')
    plt.xlabel(r'$\delta\/(\mu mol\/m^{-2} s^{-1}$)',fontsize=16)
    #plt.ylabel('f',fontsize=16)
    plt.legend(loc='upper left')
    plt.title('Random error distribution \n')
    
    ax=plt.gca()
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
def myround(x,base=10):
    return int(base*round(x/base))
    
def linear_plot():
    
    plt.subplot(212)
    
    # Create series for regression lines
    influx_x=cats_df['flux_mean'][cats_df['flux_mean']<=0]
    influx_y=np.polyval([influx_linreg_stats[0],influx_linreg_stats[1]],influx_x)
    efflux_x=cats_df['flux_mean'][cats_df['flux_mean']>=0]
    efflux_y=np.polyval([efflux_linreg_stats[0],efflux_linreg_stats[1]],efflux_x)
    
    # Do plotting
    plt.plot(cats_df['flux_mean'],cats_df['sig_del'],'ro')
    plt.plot(influx_x,influx_y,color='blue')
    plt.plot(efflux_x,efflux_y,color='blue')
    plt.xlim(round(cats_df['flux_mean'].iloc[0])-1,math.ceil(cats_df['flux_mean'].iloc[-1])+1)
    plt.xlabel(r'$C\/flux\/(\mu mol\/m^{-2} s^{-1}$)',fontsize=16)
    plt.title('Random error SD binned over flux magnitude quantiles (n='+str(num_classes)+')\n')
    
    # Move axis and relabel
    ax=plt.gca()
    str_influx=('a='+str(round(influx_linreg_stats[0],2))+
                '\nb='+str(round(influx_linreg_stats[1],2))+
                '\nr$^2$='+str(round(influx_linreg_stats[2],2))+
                '\np='+str(round(influx_linreg_stats[3],2))+
                '\nSE='+str(round(influx_linreg_stats[4],2)))
    str_efflux=('a='+str(round(efflux_linreg_stats[0],2))+
                '\nb='+str(round(efflux_linreg_stats[1],2))+
                '\nr$^2$='+str(round(efflux_linreg_stats[2],2))+
                '\np='+str(round(efflux_linreg_stats[3],2))+
                '\nSE='+str(round(efflux_linreg_stats[4],2)))
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.05, 0.35, str_influx, transform=ax.transAxes, fontsize=12,
            verticalalignment='top',bbox=props)
    ax.text(0.85, 0.35, str_efflux, transform=ax.transAxes, fontsize=12,
            verticalalignment='top',bbox=props)        
    plt.ylabel('$\sigma(\delta)$',fontsize=18)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticklabels([])
    ax2=ax.twinx()
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_position('zero')
    ax2.set_ylim(0,4.5)


###### Main program ######    

def random_error(df,records_per_day):
            
	# Do the paired difference analysis (drop all cases containing any NaN)
	diff_df=pd.DataFrame({'Fc':df['Fc']-df['Fc'].shift(records_per_day),
						  'Ta':df['Ta']-df['Ta'].shift(records_per_day),
						  'ws':df['ws']-df['ws'].shift(records_per_day),
						  'Fsd':df['Fsd']-df['Fsd'].shift(records_per_day)}).reset_index()
	diff_df.dropna(axis=0,how='any',inplace=True)
	
	return diff_df
	
	# # Print the number of tuples that pass the user-set thresholds
	# print str(len(diff_df))+' tuples passed difference constraints (S='+str(rad_threshold)+'Wm^-2, Ta='+str(temp_threshold)+'C, wind='+str(wind_threshold)+'ms^-1)\n'

	# # Cut into desired number of quantiles and make the labels a new variable in the df
	# diff_df['Category']=pd.qcut(diff_df['flux_mean'],num_classes).labels

	# # Get the flux and flux error means for each category
	# cats_df=diff_df.groupby(diff_df['Category']).mean()

	# # Use category as index for df containing flux mean and error terms
	# diff_df.index=diff_df['Category']

	# # Create sigma delta (bin-specific standard deviation of random error) variable in categories dataframe
	# cats_df['sig_del']=np.nan

	# # Calculate sigma delta for all bins
	# for i in cats_df.index:
		# cats_df['sig_del'].ix[i]=(abs(diff_df['flux_err'].ix[i]-diff_df['flux_err'].ix[i].mean())).mean()*np.sqrt(2)

	# # Calculate linear fit for +ve and -ve values...
	# influx_linreg_stats=stats.linregress(cats_df['flux_mean'][cats_df['flux_mean']<0],cats_df['sig_del'][cats_df['flux_mean']<0])
	# efflux_linreg_stats=stats.linregress(cats_df['flux_mean'][cats_df['flux_mean']>0],cats_df['sig_del'][cats_df['flux_mean']>0])

	# # Create df for propagation of error
	# prop_df=pd.DataFrame({CpropName:df[CpropName].reindex(df['Fc'].dropna().index)})
	# prop_df['sig_del']=np.where(prop_df[CpropName]>0,
								# (prop_df[CpropName]*efflux_linreg_stats[0]+efflux_linreg_stats[1])/np.sqrt(2),
								# (prop_df[CpropName]*influx_linreg_stats[0]+influx_linreg_stats[1])/np.sqrt(2))

	# # Create df to store results of random error propagation for each year
	# years_df=pd.DataFrame({'Valid_n':prop_df[CpropName].groupby([lambda x: x.year]).count()})

	# # Calculate random error for all measured data at annual time step
	# years_df['Random error ('+Cunits+')']=[random_sample_laplace(prop_df['sig_del'].ix[str(i)]) for i in years_df.index]

	# # File out
	# years_df.to_csv(os.path.join(file_out_path,file_out_name),',')

	# # Create two plot axes
	# plt.figure(1,figsize=(8,10))

	# # Plots: 1) histogram to check error distribution is approximately Laplacian;
	# #        2) linear regression - random error as function of flux magnitude (binned by user-set quantiles)
	# hist_plot()            
	# linear_plot()
	# plt.tight_layout()
	# plt.show()