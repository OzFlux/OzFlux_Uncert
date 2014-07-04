# Python modules
import numpy as np
import pandas as pd
import os

# My modules
import ustar_plots
import analysis_prep
import CPD
import CPD_QC
import CPD_stats


# Return a bootstrapped sample of the passed dataframe
def bootstrap(df):
    return df.iloc[np.random.random_integers(0,len(df)-1,len(df))]

# Prepare data, bootstrap, collate results and calculate uncertainty
def ustar_main(df,plot_out,results_out,rad_threshold,bootstrap_n,fluxfreq):
    
    reload(ustar_plots)
    reload(analysis_prep)
    reload(CPD)
    reload(CPD_QC)
    reload(CPD_stats)  
       
    # Find all years in the dataset
    years_index=df['Fc'].groupby([lambda x: x.year]).count().index
    
    interm_list=[]
    counts_list=[]
    
    ### Bootstrap the data and run the CPD algorithm
    for i in xrange(bootstrap_n):
        
        # Use the observed data as the first sample
        bootstrap_flag=(False if i==0 else True)
        
        # Bootstrap the data for each year
        if bootstrap_flag==False:
            sample_df=df
            print 'Using observational data for first pass'
        else:
            sample_df=pd.concat([bootstrap(df.ix[str(j)]) for j in years_index])
            print 'Generating bootstrap '+str(i)
        
        # Create nocturnal dataframe (drop all records where any one of the variables is NaN)
        noct_df=sample_df[['Fc','Ta','ustar']][df['Fsd']<rad_threshold].dropna(how='any',axis=0)        
        
        # Organise data into years, seasons and temperature strata, sort by temperature and u* and average across bins
        print 'Sorting data'
        try:
            years_df,seasons_df,results_df=analysis_prep.sort_data(noct_df,fluxfreq)       
        except StandardError:
            # data noct_df is problematic
            continue
        
        # Use the results df index as an iterator to run the CPD algorithm on the year/season/temperature strata
        print 'Finding change points'
        fitData_array=np.vstack([CPD.fits(seasons_df.ix[i[0]].ix[i[1]].ix[i[2]],i[0],i[1],i[2],bootstrap_flag,plot_out) for i in results_df.index])
        cols=['bMod_threshold','bMod_f_max','b1','bMod_CP','aMod_threshold','aMod_f_max','norm_a1','norm_a2','a1p','a2p']
        
        # Recompose the results dataframe with fit results
        results_df=results_df.reset_index(level=['season','T_class'],drop=True) 
        temp_S=results_df['T_avg']
        temp_index=results_df.index
        results_df=pd.DataFrame(fitData_array,columns=cols,index=temp_index)
        results_df['T_avg']=temp_S

        # QC the results and strip remaining extraneous columns
        print 'Doing QC within bootstrap'
        results_df=CPD_QC.QC1(results_df)
        
        # Run the CPD algorithm and return results
        interm_list.append(results_df)
        counts_list.append(years_df['seasons']*4)
    
    # Concatenate results (u* thresholds and counts)
    bootstrap_results_df=pd.concat(interm_list)
    output_df=pd.DataFrame({'total_count':pd.concat(counts_list).groupby(pd.concat(counts_list).index).sum()})
    
    # Do secondary QC across bootstraps
    print 'Doing QC across bootstraps'
    CPD_QC.QC2(bootstrap_results_df,output_df,bootstrap_n)
    
    # Calculate final values and add to 
    print 'Calculating final results' 
    CPD_stats.CPD_stats(bootstrap_results_df,output_df)    
            
    # Plot: 1) histograms of u* thresholds for each year; 2) normalised a1 and a2 values
    [ustar_plots.plot_hist(bootstrap_results_df['bMod_threshold'].ix[j],output_df['ustar_mean'].ix[j],output_df['ustar_sig'].ix[j],output_df['crit_t'].ix[j],j,plot_out) for j in output_df.index]
    ustar_plots.plot_slopes(output_df[['norm_a1_median','norm_a2_median']],plot_out)
    
    'Outputting data and plots'
    output_df.to_csv(os.path.join(results_out,'Results.csv'),sep=',')        
                            
    return output_df
    
#------------------------------------------------------------------------------#