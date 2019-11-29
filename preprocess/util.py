import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def roundup(x):
    return int(math.ceil(x / 10.0)) * 10

def generic_groupby(df, group, feature, agg_list):
    
    df_tem = df.groupby(group)[feature].agg(agg_list).reset_index()
    agg_list = ['std' if x==np.std else x for x in agg_list]                
    rename_dict = dict([(x,'{}_{}_{}'.format('_'.join(group), feature, x)) for x in agg_list])
    df_tem = df_tem.rename(columns = rename_dict)
    df = pd.merge(df, df_tem, how = 'left', on = group)

    return df

def applyParallel(dfGrouped, func):
    retLst = Parallel(n_jobs=8)(delayed(func)(group) for name, group in dfGrouped)
    return pd.concat(retLst)