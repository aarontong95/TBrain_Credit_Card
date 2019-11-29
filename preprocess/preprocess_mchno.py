import pandas as pd
import numpy as np


def preprocess_mchno(df):
    
    df = bacno_mchno_locdt_head_tail_diff(df)
    df = cano_days_mchno_index(df)

    return df

def bacno_mchno_locdt_head_tail_diff(df):

    df_head = df.groupby(['bacno','mchno','days']).head(1)[['bacno','mchno','days','locdt']]
    df_head = df_head.rename(columns = {'locdt' : 'locdt_head'})
    df_tail = df.groupby(['bacno','mchno','days']).tail(1)[['bacno','mchno','days','locdt']]
    df_tail = df_tail.rename(columns = {'locdt' : 'locdt_tail'})
    df_head = pd.merge(df_head, df_tail, how = 'left', on = ['bacno','mchno','days'])
    df_head['bacno_mchno_locdt_head_tail_diff'] = df_head['locdt_tail'] - df_head['locdt_head']
    df = pd.merge(df,df_head, how = 'left', on =['bacno','mchno','days'])

    return df

def cano_days_mchno_index(df):

    df['cano_days_mchno_index'] = 1
    df['cano_days_mchno_index'] = df.groupby(['cano','days','mchno'])['cano_days_mchno_index'].cumsum()
    
    return df
