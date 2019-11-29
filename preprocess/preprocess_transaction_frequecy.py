import pandas as pd
import numpy as np
from preprocess.util import generic_groupby
    
def preprocess_transaction_frequency(df):

    feature = 'txkey'
    agg_list = ['count']
    groups_list = [['cano','days'], ['cano','locdt'], ['bacno','locdt','mchno']]
    for groups in groups_list:
        df = generic_groupby(df, groups, feature, agg_list)

    return df
