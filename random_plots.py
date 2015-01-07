import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

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
def error_main(Fc_diff,cats_df,linreg_stats_df,num_classes):

    print 'Plotting: 1) PDF of random error'
    print '          2) sigma_delta (variance of random error) as a function of flux magnitude'
    fig, (ax1, ax2) = plt.subplots(nrows=2)
    fig.patch.set_facecolor('white')
    hist_plot(Fc_diff, ax1)
    linear_plot(cats_df,linreg_stats_df,num_classes,ax2)
    fig.tight_layout()
    return fig
    
def prop_main():
    return