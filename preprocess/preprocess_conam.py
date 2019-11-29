import pandas as pd
import numpy as np
from preprocess.util import generic_groupby
from preprocess.util import applyParallel


def preprocess_conam(df):

    df = preprocess_global_conam_max_min(df)
    df = diff_with_zero_conam_time(df)

    return df

def diff_with_zero_conam_time(df):

    df = df.copy()
    df_tem = df[df['conam']==0].drop_duplicates(subset = ['cano','locdt'],keep = 'first')
    df_tem = df_tem.rename(columns = {'global_time' : 'conam_zero_trans_global_time'})
    df = pd.merge(df, df_tem[['cano','locdt','conam_zero_trans_global_time']], how = 'left' , on = ['cano','locdt'])
    df['diff_gtime_with_conam_zero_trans_locdt'] = df['global_time'] - df['conam_zero_trans_global_time']
            
    return df

def preprocess_global_conam_max_min(df):
    
    group = ['cano','locdt']
    agg_list = ['min','max']
    feature = 'conam'
    df = generic_groupby(df, group, feature, agg_list)
    
    return df
