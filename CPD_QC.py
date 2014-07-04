import numpy as np
import pandas as pd

# Quality control within bootstrap

def QC1(QC1_df):

    # Set significance level (these need to be moved, and a model needs to be explicitly calculated)    
    fmax_a_threshold=6.9
    fmax_b_threshold=6.9
    
    # Find all cases that belong to minority mode (i.e. mode is sign of slope below change point)
    total_count=QC1_df['bMod_threshold'].groupby(QC1_df.index).count()
    neg_slope_count=QC1_df['bMod_threshold'][QC1_df['b1']<0].groupby(QC1_df[QC1_df['b1']<0].index).count()
    neg_slope_count=neg_slope_count.reindex(total_count.index)
    neg_slope_prop=neg_slope_count/total_count*100
    df_list=[]
    for i in neg_slope_prop.index:
        if neg_slope_prop.ix[i]<50 or pd.isnull(neg_slope_prop.ix[i]):
            df_list.append(QC1_df.ix[i]['b1']>0)
        else: 
            df_list.append(QC1_df.ix[i]['b1']<0)
    mode=pd.concat(df_list)
    mode.index=QC1_df.index
    QC1_df['major_mode']=mode
    
    # Make invalid (False) all b_model cases where: 1) fit not significantly better than null model; 2) best fit was second to last point, or;
    #                                               3) case belongs to minority mode (for that year)
    QC1_df['b_valid']=(QC1_df['bMod_f_max']>fmax_b_threshold)&(QC1_df['bMod_CP']!=48)&(QC1_df['major_mode']==True)
    
    # Make invalid (False) all a_model cases where: 1) fit not significantly better than null model; 2) slope below change point not statistically significant;
    #                                               3) slope above change point statistically significant
    QC1_df['a_valid']=(QC1_df['aMod_f_max']>fmax_a_threshold)&(QC1_df['a1p']<0.05)&(QC1_df['a2p']>0.05)
    
    # Drop the extraneous variables
    QC1_df=QC1_df.drop(['b1','bMod_f_max','bMod_CP','aMod_threshold','aMod_f_max','a1p','a2p','major_mode'],axis=1)
    
    # Return the results df
    return QC1_df

#------------------------------------------------------------------------------#

# Quality control across bootstraps

def QC2(df,QC2_df,bootstrap_n):
    
    # Get the median values of the normalised slope parameters for each year
    QC2_df['norm_a1_median']=df['norm_a1'][df['a_valid']==True].groupby(df[df['a_valid']==True].index).median()
    QC2_df['norm_a2_median']=df['norm_a2'][df['a_valid']==True].groupby(df[df['a_valid']==True].index).median()
    
    # Get the proportion of all available cases that passed QC    
    QC2_df['QCpass_count']=df['bMod_threshold'][df['b_valid']==True].groupby(df[df['b_valid']==True].index).count()
    QC2_df['QCpass_prop']=QC2_df['QCpass_count']/QC2_df['total_count']
       
    # Identify years where either diagnostic or operational model did not find enough good data for robust estimate
    QC2_df['a_valid']=(~(np.isnan(QC2_df['norm_a1_median']))&(~np.isnan(QC2_df['norm_a2_median'])))
    QC2_df['b_valid']=(QC2_df['QCpass_count']>(4*bootstrap_n))&(QC2_df['QCpass_prop']>0.2)
    for i in QC2_df.index:
        if QC2_df['a_valid'].ix[i]==False: print 'Insufficient valid cases for robust diagnostic (a model) u* determination in year '+str(i)
        if QC2_df['b_valid'].ix[i]==False: print 'Insufficient valid cases for robust operational (b model) u* determination in year '+str(i)
 
    return QC2_df    
    
#------------------------------------------------------------------------------#