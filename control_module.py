# Standard module imports
import os
import numpy as np
import pandas as pd
import xlrd
import datetime as dt
from configobj import ConfigObj
import Tkinter, tkFileDialog
import netCDF4
import sys

# Custom module imports
import ustar_threshold as ustar
import random_error

# Ingest data from the netCDF file
def get_nc(file_in):
	
    # Read file
    nc_obj=netCDF4.Dataset(file_in)
	
    # Get the time data from xlDateTime variable and turn it into a python-usable datetime object
    dates_list=[dt.datetime(*xlrd.xldate_as_tuple(elem,0)) for elem in nc_obj.variables['xlDateTime']]
	
    # Get the data from the file
    d={}
    for i in nc_obj.variables.keys():
        ndims=len(nc_obj.variables[i].shape)
        if ndims==3:
            d[i]=nc_obj.variables[i][:,0,0]
        elif ndims==1:    
            d[i]=nc_obj.variables[i][:]
    nc_obj.close()
    return pd.DataFrame(d,index=dates_list)

###### Main program ######

# Prompt user for input configuration file
root = Tkinter.Tk(); root.withdraw()
cfName = tkFileDialog.askopenfilename(initialdir='')
root.destroy()
print cfName
cf=ConfigObj(cfName)

# Set input file and output path and create directories for plots and results
file_in=os.path.join(cf['file_IO']['input_path'],cf['file_IO']['input_file'])
path_out=cf['file_IO']['output_path']
plot_path_out=os.path.join(path_out,'Plots')
if not os.path.isdir(plot_path_out): os.makedirs(os.path.join(path_out,'Plots'))
results_path_out=os.path.join(path_out,'Results')
if not os.path.isdir(results_path_out): os.makedirs(os.path.join(path_out,'Results'))

# Read input file
print 'Reading input file ...'
df=get_nc(file_in)
print 'Done!'

# Set processing flags
process_ustar=cf['processing_options']['ustar_+error']['process']=='True'
process_random_error=cf['processing_options']['random_error']['process']=='True'

# Set propagation flags
propagate_ustar_error=cf['processing_options']['ustar_+error']['propagate']=='True'
propagate_random_error=cf['processing_options']['random_error']['propagate']=='True'

# Set other parameters
radiation_threshold=int(cf['user']['global']['radiation_threshold'])
flux_frequency=int(cf['user']['global']['flux_frequency'])
error_value=int(cf['user']['global']['nan_value'])
num_trials=int(cf['user']['global']['num_MonteCarlo_trials'])

# Check dataframe for duplicates, pad as necessary and sort
df.replace(to_replace=error_value,value=np.nan,inplace=True)
df.sort(inplace=True)
df['index']=df.index
df.drop_duplicates(cols='index',take_last=True,inplace=True)
del df['index']

reload(ustar)

#------------------------------------------------------------------------------#
### u* threshold and error ###

if process_ustar:
	
    # Set variable names
    CfluxName=cf['variable_names']['ustar_+error']['carbon_flux']
    TaName=cf['variable_names']['ustar_+error']['temperature']
    ustarName=cf['variable_names']['ustar_+error']['friction_velocity']
    radName=cf['variable_names']['ustar_+error']['solar_radiation']
	
    # Set number of bootstraps
    num_bootstraps=int(cf['user']['ustar_+error']['num_bootstraps'])

    # Subset the df and create new column names
    sub_df=df[[CfluxName,TaName,ustarName,radName]]
    sub_df.columns=['Fc','Ta','ustar','Fsd']
    
    # Go do it
    ustar_results=ustar.ustar_main(sub_df,plot_path_out,results_path_out,
                                   radiation_threshold,num_bootstraps,flux_frequency)

#------------------------------------------------------------------------------#
### random error 

if process_random_error:
	
    # Set variable names
    CfluxName=cf['variable_names']['random_error']['carbon_flux']
    TaName=cf['variable_names']['random_error']['temperature']
    windspdName=cf['variable_names']['random_error']['wind_speed']
    radName=cf['variable_names']['random_error']['solar_radiation']

    # Import constraints from config file
    configs=cf['user']['random_error']
	
    # Subset the df and create new column names
    sub_df=df[[CfluxName,TaName,windspdName,radName]]
    sub_df.columns=['Fc','Ta','ws','Fsd']
	
    # Calculate statistics and output plots of random error variance
    random_results=random_error.calculate_random_error(sub_df,results_path_out,plot_path_out,configs,flux_frequency)
    
if propagate_random_error:
    
    if process_random_error:
        stats_df=random_results
    else: 
        if os.path.exists(os.path.join(results_path_out,'random_error_stats.csv')):
            stats_df=pd.read_csv(os.path.join(results_path_out,'random_error_stats.csv'))
        else:
            'Please calculate statistics of random error variance first (use random_error.calc_random_error_stats)'
            sys.exit()
            
    # Subset the df and create new column names    
    sub_df=pd.DataFrame({'Fc':df[CfluxName]})
    
    # Calculate statistics and output plots of uncertainty due to random error
    random_propagation_results=random_error.propagate_random_error(sub_df,stats_df,results_path_out,plot_path_out,num_trials)