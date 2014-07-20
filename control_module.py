# Standard module imports
import os
import numpy as np
import pandas as pd
from configobj import ConfigObj
import Tkinter, tkFileDialog

# Custom module imports
import syst_error_v26 as syst_error

###### Main program ######

# Prompt user for input configuration file
root = Tkinter.Tk(); root.withdraw()
cfName = tkFileDialog.askopenfilename(initialdir='')
root.destroy()
cf=ConfigObj(cfName)

# Set input file and read in data
file_in=os.path.join(cf['file_IO']['input_path'],cf['file_IO']['input_file'])
print 'Reading input file ...'
df=pd.read_csv(file_in,delimiter=',',parse_dates=['DT'],index_col=['DT']) 
print 'Done!'

# Set output path and create directories for plots and results
path_out=cf['file_IO']['output_path']
plot_path_out=os.path.join(path_out,'Plots')
if not os.path.isdir(plot_path_out): os.makedirs(os.path.join(file_out_path,'Plots'))
results_path_out=os.path.join(path_out,'Results')
if not os.path.isdir(results_path_out): os.makedirs(os.path.join(file_out_path,'Results'))

# Set variable names
CfluxName=cf['variable_names']['carbon_flux']
TaName=cf['variable_names']['temperature']
ustarName=cf['variable_names']['friction_velocity']
radName=cf['variable_names']['solar_radiation']

# Set processing flags
process_ustar=cf['processing_options']['ustar_threshold']

# Set other parameters
radiation_threshold=cf['user']['radiation_threshold']
num_bootstraps=cf['user']['num_bootstraps']
flux_frequency=cf['user']['flux_frequency']
records_per_day=1440/int(flux_frequency)

# Check dataframe for duplicates, pad as necessary and sort
df.sort(inplace=True)
df['index']=df.index
df.drop_duplicates(cols='index',take_last=True,inplace=True)
del df['index']

reload(syst_error)

#------------------------------------------------------------------------------#
### u* threshold and error ###

if process_ustar:

    # Subset the df and create new column names
    sub_df=df[[CfluxName,tempName,ustarName,radName]]
    sub_df.columns=['Fc','Ta','ustar','Fsd']
    
    # Go do it
    ustar_results=syst_error.ustar_main(sub_df,plot_path_out,results_path_out,
										radiation_threshold,num_bootstraps,flux_frequency)

#------------------------------------------------------------------------------#
### random error 

