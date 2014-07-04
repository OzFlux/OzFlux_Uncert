import os
import numpy as np
import pandas as pd
import syst_error_v26 as syst_error

###### Constants ######

rad_threshold=10
bootstrap_n=100


###### File IO ######

file_in_path='C:\Users\imchugh\Dropbox\Data\Site data processing\Whroo\Advanced'
file_in_name='Advanced_processed_data_Whroo_v11c.csv'
file_out_path='C:\Temp'


###### Variable names ######

CfluxName='Fc'          # CO2
tempName='Ta'           # Air temperature
ustarName='ustar'       # u*
radName='Fsd_Con'       # Incoming shortwave radiation
modName=''


###### Processing list ######

ustar_threshold=True
random_error=True
model_error=True

###### Miscellaneous ######

fluxfreq=10


###### Main program ######

# General preparations

reload(syst_error)

# Create directories for plots and results
plot_out=os.path.join(file_out_path,'Plots')
if not os.path.isdir(plot_out): os.makedirs(os.path.join(file_out_path,'Plots'))
results_out=os.path.join(file_out_path,'Results')
if not os.path.isdir(results_out): os.makedirs(os.path.join(file_out_path,'Results'))

# Set the number of records for each day
recs_day=1440/fluxfreq

# Read in the data
print 'Reading input file ...'
df=pd.read_csv(os.path.join(file_in_path,file_in_name),delimiter=',',parse_dates=['DT'],index_col=['DT']) 
print 'Done!'

# Check dataframe for duplicates, pad as necessary and sort
df.sort(inplace=True)
df['index']=df.index
df.drop_duplicates(cols='index',take_last=True,inplace=True)
del df['index']

#------------------------------------------------------------------------------#
### u* threshold and error ###

if ustar_threshold:

    # Subset the df and create new column names
    sub_df=df[[CfluxName,tempName,ustarName,radName]]
    sub_df.columns=['Fc','Ta','ustar','Fsd']
    
    # Go do it
    ustar_results=syst_error.ustar_main(sub_df,plot_out,results_out,rad_threshold,bootstrap_n,fluxfreq)

#------------------------------------------------------------------------------#

