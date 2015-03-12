# Python modules
import Tkinter, tkFileDialog
from configobj import ConfigObj
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import netCDF4
import xlrd
import datetime as dt
import ast
import numpy as np
import pandas as pd
from scipy import stats
import os
import pdb

#------------------------------------------------------------------------------
# Return a bootstrapped sample of the passed dataframe
def CPD_bootstrap(df):
    return df.iloc[np.random.random_integers(0,len(df)-1,len(df))]
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def CPD_fit(temp_df):
    
    # Only works if the index is reset here (bug?)!
    temp_df=temp_df.reset_index(drop=True)
       
    ### Calculate null model SSE for operational (b) and diagnostic (a) model
    SSE_null_b=((temp_df['Fc']-temp_df['Fc'].mean())**2).sum() # b model SSE
    alpha0,alpha1=stats.linregress(temp_df['ustar'],temp_df['Fc'])[:2] # a model regression
    SSE_null_a=((temp_df['Fc']-(temp_df['ustar']*alpha0+alpha1))**2).sum() # a model SSE
    
    ### Create empty array to hold f statistics
    f_a_array=np.empty(50)
    f_b_array=np.empty(50)
    
    # Add series to df for numpy linalg
    temp_df['int']=np.ones(50)
        
    ### Iterate through all possible change points (1-49) as below
    for i in xrange(1,49):
        
        # Operational (b) model
        temp_df['ustar_alt']=temp_df['ustar'] # Add dummy variable to df
        temp_df['ustar_alt'].iloc[i+1:]=temp_df['ustar_alt'].iloc[i]
        reg_params=np.linalg.lstsq(temp_df[['int','ustar_alt']],temp_df['Fc'])[0] # Do linear regression
        yHat=reg_params[0]+reg_params[1]*temp_df['ustar_alt'] # Calculate the predicted values for y
        SSE_full=((temp_df['Fc']-yHat)**2).sum() # Calculate SSE
        f_b_array[i]=(SSE_null_b-SSE_full)/(SSE_full/(50-2)) # Calculate and store F-score        
        
        # Diagnostic (a) model
        temp_df['ustar_alt1']=temp_df['ustar']
        temp_df['ustar_alt1'].iloc[i+1:]=temp_df['ustar_alt1'].iloc[i]
        temp_df['ustar_alt2']=(temp_df['ustar']-temp_df['ustar'].iloc[i])*np.concatenate([np.zeros(i+1),np.ones(50-(i+1))])
        reg_params=np.linalg.lstsq(temp_df[['int','ustar_alt1','ustar_alt2']],temp_df['Fc'])[0] # Do piecewise linear regression (multiple regression with dummy)          
        yHat=reg_params[0]+reg_params[1]*temp_df['ustar_alt1']+reg_params[2]*temp_df['ustar_alt2'] # Calculate the predicted values for y
        SSE_full=((temp_df['Fc']-yHat)**2).sum() # Calculate SSE
        f_a_array[i]=(SSE_null_a-SSE_full)/(SSE_full/(50-2)) # Calculate and store F-score

    # Get max f-score, associated change point and ustar value
      
    # b model
    f_b_array[0],f_b_array[-1]=f_b_array.min(),f_b_array.min()
    f_b_max=f_b_array.max()
    change_point_b=f_b_array.argmax()
    ustar_threshold_b=temp_df['ustar'].iloc[change_point_b]
   
    # a model                                                                
    f_a_array[0],f_a_array[-1]=f_a_array.min(),f_a_array.min()
    f_a_max=f_a_array.max()
    change_point_a=f_a_array.argmax()
    ustar_threshold_a=temp_df['ustar'].iloc[change_point_a]
    
    # Get regression parameters
    
    # b model
    temp_df['ustar_alt']=temp_df['ustar']
    temp_df['ustar_alt'].iloc[change_point_b+1:]=ustar_threshold_b
    reg_params=np.linalg.lstsq(temp_df[['int','ustar_alt']],temp_df['Fc'])[0]
    b0=reg_params[0]
    b1=reg_params[1]    
    
    # a model
    temp_df['ustar_alt1']=temp_df['ustar']
    temp_df['ustar_alt1'].iloc[change_point_a+1:]=temp_df['ustar_alt1'].iloc[change_point_a]
    temp_df['ustar_alt2']=(temp_df['ustar']-temp_df['ustar'].iloc[change_point_a])*np.concatenate([np.zeros(change_point_a+1),np.ones(50-(change_point_a+1))])
    reg_params=pd.ols(x=temp_df[['ustar_alt1','ustar_alt2']],y=temp_df['Fc'])
    a0=reg_params.beta.intercept
    a1=reg_params.beta.ustar_alt1
    a2=reg_params.beta.ustar_alt2
    a1p=reg_params.p_value['ustar_alt1']
    a2p=reg_params.p_value['ustar_alt2']
    norm_a1=a1*(ustar_threshold_a/(a0+a1*ustar_threshold_a))
    norm_a2=a2*(ustar_threshold_a/(a0+a1*ustar_threshold_a))

    # Return results
    return [ustar_threshold_b,f_b_max,b0,b1,change_point_b,
            ustar_threshold_a,f_a_max,a0,a1,a2,norm_a1,norm_a2,change_point_a,a1p,a2p]

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Coordinate steps in CPD process
def CPD_main():

    master_df,d=CPD_run()

    # Find number of years in df    
    years_index=list(set(master_df.index.year))
    
    # Create df to keep counts of total samples and QC passed samples
    counts_df=pd.DataFrame(index=years_index,columns=['Total'])
    counts_df.fillna(0,inplace=True)
    
    print 'Starting analysis...'    
    
    # Bootstrap the data and run the CPD algorithm
    for i in xrange(d['num_bootstraps']):
                        
        # Bootstrap the data for each year
        bootstrap_flag=(False if i==0 else True)
        if bootstrap_flag==False:
            df=master_df            
            print 'Analysing observational data for first pass'
        else:
            df=pd.concat([CPD_bootstrap(master_df.loc[str(j)]) for j in years_index])
            print 'Analysing bootstrap '+str(i)
        
        # Create nocturnal dataframe (drop all records where any one of the variables is NaN)
        temp_df=df[['Fc','Ta','ustar']][df['Fsd']<d['radiation_threshold']].dropna(how='any',axis=0)        

        # Arrange data into seasons 
        # try: may be insufficient data, needs to be handled; if insufficient on first pass then return empty,otherwise next pass
        # this will be a marginal case, will almost always be enough data in bootstraps if enough in obs data
        years_df,seasons_df,results_df=CPD_sort(temp_df,d['flux_frequency'],years_index)

        # Use the results df index as an iterator to run the CPD algorithm on the year/season/temperature strata
        print 'Finding change points...'
        cols=['bMod_threshold','bMod_f_max','b0','b1','bMod_CP',
              'aMod_threshold','aMod_f_max','a0','a1','a2','norm_a1','norm_a2','aMod_CP','a1p','a2p']
        stats_df=pd.DataFrame(np.vstack([CPD_fit(seasons_df.loc[j]) for j in results_df.index]),                      
                              columns=cols,index=results_df.index)
        results_df=results_df.join(stats_df)        
        print 'Done!'
        
        results_df['bMod_CP']=results_df['bMod_CP'].astype(int)
        results_df['aMod_CP']=results_df['aMod_CP'].astype(int)
        
        # QC the results
        print 'Doing within-sample QC...'
        results_df=CPD_QC1(results_df)
        print 'Done!' 
        
        # Output results and plots (if user has set output flags in config file to true)
        if bootstrap_flag==False:
            if 'results_output_path' in d.keys(): 
                print 'Outputting results for all years / seasons / T classes in observational dataset'
                results_df.to_csv(os.path.join(d['results_output_path'],'Observational_ustar_threshold_statistics.csv'))
            if 'plot_output_path' in d.keys(): 
                print 'Doing plotting for observational data'
                for j in results_df.index:
                    CPD_plot_fits(seasons_df.loc[j],results_df.loc[j],d['plot_output_path'])
        
        # Drop the season and temperature class levels from the hierarchical index, 
        # drop all cases that failed QC
        results_df=results_df.reset_index(level=['season','T_class'],drop=True)
        results_df=results_df[results_df['b_valid']==True]
        
        # If first pass, create a df to concatenate the results for each individual run
        # Otherwise concatenate all_results_df with current results_df
        if bootstrap_flag==False:
            all_results_df=results_df
        else:
            all_results_df=pd.concat([all_results_df,results_df])
        
        # Iterate counters for each year for each bootstrap
        for i in years_df.index:
            counts_df['Total'].loc[i]=counts_df['Total'].loc[i]+years_df['seasons'].loc[i]*4
    
    print 'Finished change point detection for all bootstraps'
    print 'Starting QC'    
    
    # Sort by index so all years are together
    all_results_df.sort_index(inplace=True)
    
    # Drop all years with no data remaining after QC, and return nothing if all years were dropped
    [counts_df.drop(i,inplace=True) for i in counts_df.index if counts_df['Total'].loc[i]==0]    
    if counts_df.empty:
        print 'Insufficient data for analysis... exiting'
        return
    
    # QC the combined results
    print 'Doing cross-sample QC...'
    output_stats_df=CPD_QC2(all_results_df,counts_df,d['num_bootstraps'])
    print 'Done!' 

    # Calculate final values
    print 'Calculating final results' 
    output_stats_df=CPD_stats(all_results_df,output_stats_df)    
    
    # If requested by user, plot: 1) histograms of u* thresholds for each year; 
    #                             2) normalised a1 and a2 values
    if 'plot_output_path' in d.keys():
        print 'Plotting u* histograms for all valid b model thresholds for all valid years'
        [CPD_plot_hist(all_results_df['bMod_threshold'].loc[j][all_results_df['b_valid'].loc[j]==True],
                       output_stats_df['ustar_mean'].loc[j],
                       output_stats_df['ustar_sig'].loc[j],
                       output_stats_df['crit_t'].loc[j],
                       j, d['plot_output_path'])
         for j in output_stats_df.index]
        
        print 'Plotting normalised median slope parameters for all valid a model thresholds for all valid years'
        CPD_plot_slopes(output_stats_df[['norm_a1_median','norm_a2_median']],d['plot_output_path'])    
    
    # Output final stats if requested by user
    if 'results_output_path' in d.keys():
        'Outputting final results'
        output_stats_df.to_csv(os.path.join(d['results_output_path'],'annual_statistics.csv'))    
    
    print 'Analysis complete!'
    # Return final results
    return output_stats_df    
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Plot identified change points in observed (i.e. not bootstrapped) data and   
# write to specified folder                                                    
def CPD_plot_fits(temp_df,stats_df,plot_out):
    
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
def CPD_plot_hist(S,mu,sig,crit_t,year,plot_out):
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
def CPD_plot_slopes(df,plot_out):
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
def CPD_QC1(QC1_df):
    
    # Set significance level (these need to be moved, and a model needs to be explicitly calculated for a threshold)    
    fmax_a_threshold=6.9
    fmax_b_threshold=6.9
    
    QC1_df['major_mode']=True
    
    # For each year, find all cases that belong to minority mode (i.e. mode is sign of slope below change point)
    total_count=QC1_df['bMod_threshold'].groupby(level='year').count()
    neg_slope=QC1_df['bMod_threshold'][QC1_df['b1']<0].groupby(level='year').count()
    neg_slope=neg_slope.reindex(total_count.index)
    neg_slope=neg_slope.fillna(0)
    neg_slope=neg_slope/total_count*100
    for i in neg_slope.index:
        sign=1 if neg_slope.loc[i]<50 else -1
        QC1_df['major_mode'].loc[i]=np.sign(np.array(QC1_df['b1'].loc[i]))==sign
    
    # Make invalid (False) all b_model cases where: 1) fit not significantly better than null model; 
    #                                               2) best fit at extreme ends;
    #                                               3) case belongs to minority mode (for that year)
    QC1_df['b_valid']=((QC1_df['bMod_f_max']>fmax_b_threshold)
                       &(QC1_df['bMod_CP']>4)
                       &(QC1_df['bMod_CP']<45)
                       &(QC1_df['major_mode']==True))
        
    # Make invalid (False) all a_model cases where: 1) fit not significantly better than null model; 
    #                                               2) slope below change point not statistically significant;
    #                                               3) slope above change point statistically significant
    QC1_df['a_valid']=((QC1_df['aMod_f_max']>fmax_a_threshold)
                       &(QC1_df['a1p']<0.05)&(QC1_df['a2p']>0.05))
    
    # Return the results df
    QC1_df=QC1_df.drop('major_mode',axis=1)
    return QC1_df
    
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Quality control across bootstraps
def CPD_QC2(df,output_df,bootstrap_n):
    
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
        if output_df['a_valid'].loc[i]==False: print 'Insufficient valid cases for robust diagnostic (a model) u* determination in year '+str(i)
        if output_df['b_valid'].loc[i]==False: print 'Insufficient valid cases for robust operational (b model) u* determination in year '+str(i)
 
    return output_df    
    
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Fetch the data and prepare it for analysis
def CPD_run():
        
    # Prompt user for configuration file and get it
    root = Tkinter.Tk(); root.withdraw()
    cfName = tkFileDialog.askopenfilename(initialdir='')
    root.destroy()
    cf=ConfigObj(cfName)
    
    # Set input file and output path and create directories for plots and results
    file_in=os.path.join(cf['files']['input_path'],cf['files']['input_file'])
    path_out=cf['files']['output_path']
    plot_path_out=os.path.join(path_out,'Plots')
    if not os.path.isdir(plot_path_out): os.makedirs(os.path.join(path_out,'Plots'))
    results_path_out=os.path.join(path_out,'Results')
    if not os.path.isdir(results_path_out): os.makedirs(os.path.join(path_out,'Results'))    
    
    # Get user-set variable names from config file
    vars_data=[cf['variables']['data'][i] for i in cf['variables']['data']]
    vars_QC=[cf['variables']['QC'][i] for i in cf['variables']['QC']]
    vars_all=vars_data+vars_QC
       
    # Read .nc file
    nc_obj=netCDF4.Dataset(file_in)
    flux_frequency=int(nc_obj.time_step)
    dates_list=[dt.datetime(*xlrd.xldate_as_tuple(elem,0)) for elem in nc_obj.variables['xlDateTime']]
    d={}
    for i in vars_all:
        ndims=len(nc_obj.variables[i].shape)
        if ndims==3:
            d[i]=nc_obj.variables[i][:,0,0]
        elif ndims==1:    
            d[i]=nc_obj.variables[i][:]
    nc_obj.close()
    df=pd.DataFrame(d,index=dates_list)    
        
    # Build dictionary of additional configs
    d={}
    d['radiation_threshold']=int(cf['options']['radiation_threshold'])
    d['num_bootstraps']=int(cf['options']['num_bootstraps'])
    d['flux_frequency']=flux_frequency
    if cf['options']['output_plots']=='True':
        d['plot_output_path']=plot_path_out
    if cf['options']['output_results']=='True':
        d['results_output_path']=results_path_out
        
    # Replace configured error values with NaNs and remove data with unacceptable QC codes, then drop flags
    df.replace(int(cf['options']['nan_value']),np.nan)
    if 'QC_accept_codes' in cf['options']:    
        QC_accept_codes=ast.literal_eval(cf['options']['QC_accept_codes'])
        eval_string='|'.join(['(df[vars_QC[i]]=='+str(i)+')' for i in QC_accept_codes])
        for i in xrange(4):
            df[vars_data[i]]=np.where(eval(eval_string),df[vars_data[i]],np.nan)
    df=df[vars_data]
    
    return df,d
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def CPD_sort(df,fluxfreq,years_index):
    
    # Set the bin size on the basis of the flux measurement frequency
    if fluxfreq==30:
        bin_size=1000
    else:
        bin_size=600
    
    # Create a df containing count stats for the variables for all available years
    years_df=pd.DataFrame(index=years_index)
    years_df['Fc_count']=df['Fc'].groupby([lambda x: x.year]).count()
    years_df['seasons']=[years_df['Fc_count'].loc[j]/(bin_size/2)-1 for j in years_df.index]
    years_df['seasons'].fillna(0,inplace=True)
    years_df['seasons']=np.where(years_df['seasons']<0,0,years_df['seasons'])
    years_df['seasons']=years_df['seasons'].astype(int)
    if np.all(years_df['seasons']<=0):
        print 'No years with sufficient data for evaluation. Exiting...'
        return
    elif np.any(years_df['seasons']<=0):
        exclude_years_list=years_df[years_df['seasons']<=0].index.tolist()
        exclude_years_str= ','.join(map(str,exclude_years_list))
        print 'Insufficient data for evaluation in the following years: '+exclude_years_str+' (excluded from analysis)'
        years_df=years_df[years_df['seasons']>0]
    
    # Extract overlapping series, sort by temperature and concatenate
    # Note .ix is required for some reason; .loc won't work (may be a bug)
    lst=[]
    for year in years_df.index:
        for season in xrange(years_df['seasons'].loc[year]):
            start_ind=season*(bin_size/2)
            end_ind=season*(bin_size/2)+bin_size
            lst.append(df.ix[str(year)].iloc[start_ind:end_ind].sort('Ta',axis=0))
    seasons_df=pd.concat([dataframe for dataframe in lst])

    # Make a hierarchical index for year, season, temperature class, bin for the seasons dataframe
    # Do year
    years_index=np.concatenate([np.int32(np.ones(years_df['seasons'].loc[year]*bin_size)*year) 
                                for year in years_df.index])
    
    # Do season                                
    seasons_index=np.concatenate([np.concatenate([np.int32(np.ones(bin_size)*(season+1)) 
                                                  for season in xrange(years_df['seasons'].loc[year])]) 
                                                  for year in years_df.index])

    # Do temperature class    
    Tclass_index=np.tile(np.concatenate([np.int32(np.ones(bin_size/4)*(i+1)) for i in xrange(4)]),
                         len(seasons_index)/bin_size)
    
    # Do bin
    bin_index=np.tile(np.int32(np.arange(bin_size/4)/(bin_size/200)),len(seasons_df)/(bin_size/4))
    arrays=[years_index,seasons_index,Tclass_index]
    tuples=list(zip(*arrays))
    hierarchical_index=pd.MultiIndex.from_tuples(tuples,names=['year','season','T_class'])
    seasons_df.index=hierarchical_index
    
    # Set up the results df, sort the seasons by ustar, then bin average and drop the bin level from the index
    results_df=pd.DataFrame({'T_avg':seasons_df['Ta'].groupby(level=['year','season','T_class']).mean()})
    seasons_df=pd.concat([seasons_df.loc[i[0]].loc[i[1]].loc[i[2]].sort('ustar',axis=0) for i in results_df.index])
    seasons_df.index=hierarchical_index
    seasons_df=seasons_df.set_index(bin_index,append=True)
    seasons_df.index.names=['year','season','T_class','bin']
    seasons_df=seasons_df.groupby(level=['year','season','T_class','bin']).mean()
    seasons_df=seasons_df.reset_index(level=['bin'],drop=True)
    seasons_df=seasons_df[['ustar','Fc']]
    
    return years_df,seasons_df,results_df
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def CPD_stats(df,stats_df):
    
    # Add statistics vars to output df
    stats_df['ustar_mean']=np.nan
    stats_df['ustar_sig']=np.nan
    stats_df['ustar_n']=np.nan
    stats_df['crit_t']=np.nan
    stats_df['95%CI_lower']=np.nan
    stats_df['95%CI_upper']=np.nan
    stats_df['skew']=np.nan
    stats_df['kurt']=np.nan
        
    # Drop data that failed b model, then drop b model boolean variable
    df=df[df['b_valid']==True]
    df=df.drop('b_valid',axis=1)
    
    # Calculate stats
    for i in stats_df.index:
        if stats_df['b_valid'].loc[i]:
            if isinstance(df['bMod_threshold'].loc[i],pd.Series):
                temp=stats.describe(df['bMod_threshold'].loc[i])
                stats_df['ustar_mean'].loc[i]=temp[2]
                stats_df['ustar_sig'].loc[i]=np.sqrt(temp[3])
                stats_df['crit_t'].loc[i]=stats.t.ppf(1-0.025,temp[0])
                stats_df['95%CI_lower'].loc[i]=stats_df['ustar_mean'].loc[i]-stats_df['ustar_sig'].loc[i]*stats_df['crit_t'].loc[i]
                stats_df['95%CI_upper'].loc[i]=stats_df['ustar_mean'].loc[i]+stats_df['ustar_sig'].loc[i]*stats_df['crit_t'].loc[i]
                stats_df['skew'].loc[i]=temp[4]
                stats_df['kurt'].loc[i]=temp[5]
            else:
                stats_df['ustar_mean'].loc[i]=df['bMod_threshold'].loc[i]
            
    return stats_df
#------------------------------------------------------------------------------
    
if __name__=='__main__':
    test=CPD_main()