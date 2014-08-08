# Python modules
import Tkinter, tkFileDialog
from configobj import ConfigObj
import netCDF4
import xlrd
import datetime as dt
import ast
import numpy as np
import pandas as pd
from scipy import stats
import pdb
import os

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
        d[i]=nc_obj.variables[i][:]
    nc_obj.close()
    df=pd.DataFrame(d,index=dates_list)    
        
    # Build dictionary of additional configs
    d={}
    d['radiation_threshold']=int(cf['options']['radiation_threshold'])
    d['num_bootstraps']=int(cf['options']['num_bootstraps'])
    d['flux_frequency']=flux_frequency
        
    # Replace configured error values with NaNs and remove data with unacceptable QC codes, then drop flags
    df.replace(int(cf['options']['nan_value']),np.nan)
    if 'QC_accept_codes' in cf['options']:    
        QC_accept_codes=ast.literal_eval(cf['options']['QC_accept_codes'])
        eval_string='|'.join(['(df[vars_QC[i]]=='+str(i)+')' for i in QC_accept_codes])
        for i in xrange(4):
            df[vars_data[i]]=np.where(eval(eval_string),df[vars_data[i]],np.nan)
    df=df[vars_data]
    
    return df,d

    # Return a bootstrapped sample of the passed dataframe
def bootstrap(sample_df):
    return sample_df.iloc[np.random.random_integers(0,len(sample_df)-1,len(sample_df))]

def CPD_main():

    df,d=CPD_run()
    
    ### Bootstrap the data and run the CPD algorithm
    for i in xrange(bootstrap_n):
        
        # Use the observed data as the first sample
        bootstrap_flag=(False if i==0 else True)
        
        # Bootstrap the data for each year
        if bootstrap_flag==False:
            print 'Using observational data for first pass'
        else:
            df=pd.concat([bootstrap(df.ix[str(j)]) for j in years_index])
            print 'Generating bootstrap '+str(i)
        
        # Create nocturnal dataframe (drop all records where any one of the variables is NaN)
        df=df[['Fc','Ta','ustar']][df['Fsd']<rad_threshold].dropna(how='any',axis=0)        
        
#                
#        
#        # Organise data into years, seasons and temperature strata, sort by temperature and u* and average across bins
#        print 'Sorting data'
#        try:
#            years_df,seasons_df,results_df=analysis_prep.sort_data(noct_df,fluxfreq)       
#        except StandardError:
#            # data noct_df is problematic
#            continue
#        
#        # Use the results df index as an iterator to run the CPD algorithm on the year/season/temperature strata
#        print 'Finding change points'
#        fitData_array=np.vstack([CPD.fits(seasons_df.ix[i[0]].ix[i[1]].ix[i[2]],i[0],i[1],i[2],bootstrap_flag,plot_out) for i in results_df.index])
#        cols=['bMod_threshold','bMod_f_max','b1','bMod_CP','aMod_threshold','aMod_f_max','norm_a1','norm_a2','a1p','a2p']
#        
#        # Recompose the results dataframe with fit results
#        results_df=results_df.reset_index(level=['season','T_class'],drop=True) 
#        temp_S=results_df['T_avg']
#        temp_index=results_df.index
#        results_df=pd.DataFrame(fitData_array,columns=cols,index=temp_index)
#        results_df['T_avg']=temp_S
#
#        # QC the results and strip remaining extraneous columns
#        print 'Doing QC within bootstrap'
#        results_df=CPD_QC.QC1(results_df)
#        
#        # Run the CPD algorithm and return results
#        interm_list.append(results_df)
#        counts_list.append(years_df['seasons']*4)
    
# Find change point for model with slope above change point (Barr's
# 'diagnostic' model)

def fits(temp_df,year,season,T_class,bootstrap_flag,plot_out_path):
    
    # Only works if the index is reset here (bug?)!
    temp_df=temp_df.reset_index(drop=True)
       
    ### Calculate null model SSE for operational (b) and diagnostic (a) model
    SSE_null_b=((temp_df['Fc']-temp_df['Fc'].mean())**2).sum() # b model SSE
    alpha0,alpha1=stats.linregress(temp_df['ustar'],temp_df['ustar'])[:2] # a model regression
    SSE_null_a=((temp_df['Fc']-(temp_df['ustar']*alpha0+alpha1))**2).sum() # a model SSE
    
    ### Create empty array to hold f statistics
    f_a_array=np.empty(50)
    f_b_array=np.empty(50)
    
    # Add series to df for OLS
    temp_df['int']=np.ones(50)
    
    ### Iterate through all possible change points (1-49) as below
    for i in xrange(1,49):
        
        # Operational model
        temp_df['ustar_alt']=temp_df['ustar'] # Add dummy variable to df
        temp_df['ustar_alt'].iloc[i+1:]=temp_df['ustar_alt'].iloc[i]
        reg_params=np.linalg.lstsq(temp_df[['int','ustar_alt']],temp_df['Fc'])[0] # Do linear regression
        yHat=reg_params[0]+reg_params[1]*temp_df['ustar_alt'] # Calculate the predicted values for y
        SSE_full=((temp_df['Fc']-yHat)**2).sum() # Calculate SSE
        f_b_array[i]=(SSE_null_b-SSE_full)/(SSE_full/(50-2)) # Calculate and store F-score        
        
        # Diagnostic model
        temp_df['dummy']=(temp_df['ustar']-temp_df['ustar'].iloc[i])*np.concatenate([np.zeros(i+1),np.ones(50-(i+1))]) # Add dummy variable to df
        reg_params=np.linalg.lstsq(temp_df[['int','ustar','dummy']],temp_df['Fc'])[0] # Do piecewise linear regression (multiple regression with dummy)          
        yHat=reg_params[0]+reg_params[1]*temp_df['ustar']+reg_params[2]*temp_df['dummy'] # Calculate the predicted values for y
        SSE_full=((temp_df['Fc']-yHat)**2).sum() # Calculate SSE
        f_a_array[i]=(SSE_null_a-SSE_full)/(SSE_full/(50-2)) # Calculate and store F-score
        
    ### Pad f-score array and get maximum value and associated change point and ustar value
    
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
    
    ### Do stats
    
    # b model
    temp_df['ustar_alt']=temp_df['ustar']
    temp_df['ustar_alt'].iloc[change_point_b+1:]=ustar_threshold_b
    reg_params=np.linalg.lstsq(temp_df[['int','ustar_alt']],temp_df['Fc'])[0]
    b0=reg_params[0]
    b1=reg_params[1]
        
    # a model (note that ols has to be done twice because pandas ols returns strange result - at least different to numpy - for intercept)
    temp_df['dummy']=(temp_df['ustar']-ustar_threshold_a)*np.concatenate([np.zeros(change_point_a+1),np.ones(50-(change_point_a+1))])
    reg_params=np.linalg.lstsq(temp_df[['int','ustar','dummy']],temp_df['Fc'])[0]
    a0=reg_params[0]
    a1=reg_params[1]
    a2=reg_params[2]     
    reg_params=pd.ols(x=temp_df[['int','ustar','dummy']],y=temp_df['Fc']) 
    a1p=reg_params.p_value['ustar']
    a2p=reg_params.p_value['dummy']
    norm_a1=a1*(ustar_threshold_a/(a0+a1*ustar_threshold_a))
    norm_a2=a2*(ustar_threshold_a/(a0+a1*ustar_threshold_a))
    
    ### Call plotting (pass obs data to plotting function)
    if bootstrap_flag==False:
        temp_df['yHat_b']=b0+b1*temp_df['ustar_alt']
        temp_df['yHat_a']=a0+a1*temp_df['ustar']+a2*temp_df['dummy'] # Calculate the estimated time series                                                                                                              
        ustar_plots.plot_fits(temp_df,ustar_threshold_b,change_point_b,year,season,T_class,plot_out_path)    
    
    ### Return the temporary df
    return [ustar_threshold_b,f_b_max,b1,change_point_b,ustar_threshold_a,f_a_max,norm_a1,norm_a2,a1p,a2p]

#------------------------------------------------------------------------------#