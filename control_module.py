# Standard module imports
import os
import numpy as np
import pandas as pd
import xlrd
import datetime as dt
from configobj import ConfigObj
import Tkinter, tkFileDialog
import netCDF4

# Custom module imports
import ustar_threshold as ustar

def get_nc(file_in):
	
    # Read file
    nc_obj=netCDF4.Dataset(file_in)
	
    # Get the time data from xlDateTime variable and turn it into a python-usable datetime object
    dates_list=[dt.datetime(*xlrd.xldate_as_tuple(elem,0)) for elem in nc_obj.variables['xlDateTime']]
	
    # Get the data from the file
    d={}
    for i in nc_obj.variables.keys():
        d[i]=nc_obj.variables[i][:]
    nc_obj.close()
    return pd.DataFrame(d,index=dates_list)

###### Main program ######

# Prompt user for input configuration file
root = Tkinter.Tk(); root.withdraw()
cfName = tkFileDialog.askopenfilename(initialdir='')
root.destroy()
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

# Set variable names
CfluxName=cf['variable_names']['carbon_flux']
TaName=cf['variable_names']['temperature']
ustarName=cf['variable_names']['friction_velocity']
radName=cf['variable_names']['solar_radiation']

# Set processing flags
process_ustar=cf['processing_options']['ustar_threshold']
process_random=cf['processing_options']['random_error']

# Set other parameters
radiation_threshold=int(cf['user']['radiation_threshold'])
flux_frequency=int(cf['user']['flux_frequency'])
records_per_day=1440/int(flux_frequency)

# Check dataframe for duplicates, pad as necessary and sort
df.sort(inplace=True)
df['index']=df.index
df.drop_duplicates(cols='index',take_last=True,inplace=True)
del df['index']

reload(ustar)

#------------------------------------------------------------------------------#
### u* threshold and error ###

if process_ustar:
	
	# Set number of bootstraps
	num_bootstraps=int(cf['user']['num_bootstraps'])

    # Subset the df and create new column names
	sub_df=df[[CfluxName,TaName,ustarName,radName]]
	sub_df.columns=['Fc','Ta','ustar','Fsd']
    
    # Go do it
	ustar_results=ustar.ustar_main(sub_df,plot_path_out,results_path_out,
								   radiation_threshold,num_bootstraps,flux_frequency)

#------------------------------------------------------------------------------#
### random error 

