# My modules
import ustar_plots
import numpy as np
import pandas as pd
from scipy import stats
import pdb
import os

# Find change point for model with slope above change point (Barr's
# 'diagnostic' model)

def fits(temp_df,year,season,T_class,bootstrap_flag,plot_out_path):
    
    # Only works if the index is reset here (bug?)!
    temp_df=temp_df.reset_index(drop=True)
   
    #if (year==2011)&(season==3)&(T_class==3):
    #    temp_df.to_pickle(os.path.join('C:\Temp\Results','temp.df'))        
       
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