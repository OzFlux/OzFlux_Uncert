import numpy as np
from scipy import stats
import pandas as pd

# Calculate statistics for ustar values

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
        if isinstance(df['bMod_threshold'].ix[i],pd.Series):
            temp=stats.describe(df['bMod_threshold'].ix[i])
            stats_df['ustar_mean'].ix[i]=temp[2]
            stats_df['ustar_sig'].ix[i]=np.sqrt(temp[3])
            stats_df['ustar_n']
            stats_df['crit_t'].ix[i]=stats.t.ppf(1-0.025,temp[0])
            stats_df['95%CI_lower'].ix[i]=stats_df['ustar_mean'].ix[i]-stats_df['ustar_sig'].ix[i]*stats_df['crit_t'].ix[i]
            stats_df['95%CI_upper'].ix[i]=stats_df['ustar_mean'].ix[i]+stats_df['ustar_sig'].ix[i]*stats_df['crit_t'].ix[i]
            stats_df['skew'].ix[i]=temp[4]
            stats_df['kurt'].ix[i]=temp[5]
        else:
            stats_df['ustar_mean'].ix[i]=df['bMod_threshold'].ix[i]
#------------------------------------------------------------------------------# 