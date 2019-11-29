import pandas as pd
import numpy as np


def preprocess_train_test_split(df, cat_features):

    ''' 
        Replace the value of categorical features of training set with NA 
        if the value is not in testing set
    '''

    df_train = df[~df['fraud_ind'].isna()]
    df_test = df[df['fraud_ind'].isna()]

    # Keep the original value of some feature before replacing
    keep_list = ['cano',
                 'bacno',
                 'mchno'
                 ]
    for keep in keep_list:
        df_train[keep + '_original'] = df_train[keep].copy()
    
    for feature in cat_features:
        test_unique = df_test[feature].unique()
        df_train[feature] = np.where(df_train[feature].isin(test_unique), df_train[feature], np.nan)

    df_test[cat_features] = df_test[cat_features].astype('category')
    df_train[cat_features] = df_train[cat_features].astype('category')

    return df_train, df_test
