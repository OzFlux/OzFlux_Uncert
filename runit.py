#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 15:25:39 2020

@author: imchugh
"""

import xarray as xr
import cpd

# Set directories for source and output data
data_path = '/home/unimelb.edu.au/imchugh/Desktop/GatumPasture_L3.nc'
results_path = '/home/unimelb.edu.au/imchugh/Desktop'

##############################################################################
### OPTION 1 #################################################################
##############################################################################

# Get the data and convert to dataframe
ds = xr.open_dataset(data_path)
df = (ds[['ustar', 'Fc', 'Fsd', 'Ta']].sel(latitude=ds.latitude[0],
                                           longitude=ds.longitude[0],
                                           drop=True)
      .to_dataframe()
      )

# Create instance of cpd class and run get_change_points method, setting a path
# for writing of data to excel output file (optional)
x_0 = cpd.change_point_detect(df)
results_dict_0 = x_0.get_change_points(n_trials=3, write_to_dir=results_path)

##############################################################################
### OPTION 2 #################################################################
##############################################################################

# Create instance of cpd class and run get_change_points method, setting a path
# for writing of data to excel output file (optional)
x_1 = cpd.change_point_detect_from_netcdf(data_path)
results_dict_1 = x_1.get_change_points(n_trials=3, write_to_dir=results_path)

# Note that if you want to run single years, just set the 'years' kwarg to a
# list of years (must be a list of either str or int) eg:
results_dict_2 = x_1.get_change_points(n_trials=3, years=[2017])

# And if you want to write to the attrs of an nc file, use 'write_to_nc' method
# of change_point_detect_from_netcdf class (note this is only available in
# this class, and has no options other than n_trials, it just runs all eligible
# years in the data then writes a string of years and mean ustar change point
# values to a global attr called 'ustar_thresholds'), eg:
results_dict_3 = x_1.write_change_points_to_nc(n_trials=3)