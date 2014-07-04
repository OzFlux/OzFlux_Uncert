import numpy as np
import pandas as pd
import sys
import pdb


# Create arrays of appropriate length for each of year, season, temperature
# class and averaging bin and convert to hierarchical index

def make_working_index(years_df,bin_size):
       
    years_index=np.concatenate([np.int32(np.ones(years_df['seasons'].ix[i]*bin_size)*i) 
                                for i in years_df.index])
    seasons_index=np.concatenate([np.concatenate([np.int32(np.ones(bin_size)*(i+1)) 
                                  for i in xrange(years_df['seasons'].ix[j])]) for j in years_df.index])
    Tclass_index=np.tile(np.concatenate([np.int32(np.ones(bin_size/4)*(i+1)) 
                                         for i in xrange(4)]),len(seasons_index)/bin_size)
    arrays=[years_index,seasons_index,Tclass_index]
    tuples=list(zip(*arrays))
    return pd.MultiIndex.from_tuples(tuples,names=['year','season','T_class'])

#------------------------------------------------------------------------------#

# Split series into overlapping seasons

def sort_data(df,fluxfreq):
    
    # Set the bin size on the basis of the flux measurement frequency
    if fluxfreq==30:
        bin_size=1000
    else:
        bin_size=600
    
    # Create a df containing count stats for the variables for all available years
    years_df=pd.DataFrame({'Fc_count':df['Fc'].groupby([lambda x: x.year]).count()})

    # Calculate the number of overlapping pseudo-seasons available for the year 
    # (4 T classes with 50 bins of 5 obs [= 250 total obs] in each class [= 1000 total obs] overlapping by 50%)
    # (4 T classes with 50 bins of 3 obs [= 150 total obs] in each class [= 600 total obs] overlapping by 50%)    
    years_df['seasons']=[years_df['Fc_count'].ix[j]/(bin_size/2)-1 for j in years_df.index]  
    
    # If there is not enough data:
    # - in any year then quit
    # - in some years then drop those years from the subsequent analysis
    if not np.any(years_df['seasons']):
        print 'No years with sufficient data for evaluation. Exiting...'
        sys.exit()
    elif not np.all(years_df['seasons']):
        exclude_years_list=years_df[years_df['seasons']==0].index.tolist()
        exclude_years_str= ','.join(map(str,exclude_years_list))
        print 'Insufficient data for evaluation in the following years: '+exclude_years_str+' (excluded from analysis)'
        years_df=years_df[years_df['seasons']>0]
    
    # Extract overlapping series, sort by temperature and concatenate
    try:
        seasons_df=pd.concat([pd.concat([df.ix[str(i)].iloc[j*(bin_size/2):j*(bin_size/2)+bin_size].sort('Ta',axis=0) 
                                        for j in xrange(years_df['seasons'].ix[i])]) for i in years_df.index])
    except:
        pdb.set_trace()
        raise StandardError
        
    
    # Create hierarchical index and apply
    seasons_index=make_working_index(years_df,bin_size)
    seasons_df.index=seasons_index
    
    # Average the data to get T classes and set up the results df
    results_df=pd.DataFrame({'T_avg':seasons_df['Ta'].groupby(level=['year','season','T_class']).mean()})
    
    # Sort each temperature stratum (for each season and year) by ustar and reassemble                                                
    seasons_df=pd.concat([seasons_df.ix[i[0]].ix[i[1]].ix[i[2]].sort('ustar',axis=0) for i in results_df.index])
    seasons_df.index=seasons_index

    # Create bins for averaging
    seasons_df=seasons_df.set_index(np.tile(np.int32(np.arange(bin_size/4)/(bin_size/200)),len(seasons_df)/(bin_size/4)),append=True)
    seasons_df.index.names=['year','season','T_class','bin']
    
    # Average the data across bins and drop the bin level in the Multi-index
    seasons_df=seasons_df.groupby(level=['year','season','T_class','bin']).mean()
    seasons_df=seasons_df.reset_index(level=['bin'],drop=True)

    # Drop the temperature variable
    seasons_df=seasons_df[['ustar','Fc']]
   
    return years_df,seasons_df,results_df

#------------------------------------------------------------------------------# 