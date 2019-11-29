import pandas as pd
import numpy as np
from preprocess.util import generic_groupby
    
def preprocess_time(df):

    df['global_time'] = loctm_to_global_time(df)
    df['last_time_days'] = df.groupby(['cano','days'])['global_time'].diff(periods = 1)
    df['next_time_days'] = df.groupby(['cano','days'])['global_time'].diff(periods = -1)

    groups = ['cano','locdt']
    feature = 'global_time'
    agg_list = [np.std]
    df = generic_groupby(df, groups, feature, agg_list)
    
    return df

def loctm_to_global_time(df):
    
    df = df.copy()
    df['loctm'] = df['loctm'].astype(str)
    df['loctm'] = df['loctm'].str[:-2]
    df['hours'] = df['loctm'].str[-6:-4]
    df['hours'] = np.where(df['hours']=='', '0', df['hours']).astype(int)
    df['minutes'] = df['loctm'].str[-4:-2]
    df['minutes'] = np.where(df['minutes']=='', '0', df['minutes']).astype(int)
    df['second'] = df['loctm'].str[-2:].astype(int)
    df['loctm'] = df['hours']*60*60 + df['minutes']*60 + df['second']
    df['global_time'] = df['locdt']*24*60*60 + df['hours']*60*60+df['minutes']*60+df['second']
                        
    return df['global_time']