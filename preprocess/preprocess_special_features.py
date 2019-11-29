import pandas as pd
import numpy as np

def preprocess_special_features(df):

    df = diff_with_first_fraud_locdt(df) 
    df = black_white_list(df) 

    return df

def black_white_list(df):

    df[['mchno','conam']] = df[['mchno','conam']].astype(str)
    df['normal_mchno'] = df.apply(lambda x : x['mchno'] if x['fraud_ind']==0 else -999,axis = 1)
    df['fraud_mchno'] = df.apply(lambda x : x['mchno'] if x['fraud_ind']==1 else -999,axis = 1)
    df['fraud_conam'] = df.apply(lambda x : x['conam'] if x['fraud_ind']==1 else -999,axis = 1)

    rolling_list = ['normal_mchno',
                    'fraud_mchno',
                    'fraud_conam'
                     ]

    for feature in rolling_list:
        df[feature] = df[feature].apply(lambda x : [x])
        df['rolling_{}'.format(feature)] = df.groupby('bacno')[feature].apply(lambda x : x.cumsum())

        df_tem = df.drop_duplicates(subset = ['cano','locdt'],keep = 'last')
        df_tem['last_rolling_{}_cano'.format(feature)] = df_tem.groupby(['cano'])['rolling_{}'.format(feature)].shift(1)
        df = pd.merge(df, df_tem[['cano','locdt','last_rolling_{}_cano'.format(feature)]], how = 'left', on = ['cano','locdt'])
        df['last_rolling_{}_cano'.format(feature)] = df['last_rolling_{}_cano'.format(feature)].fillna('NA')

        df['{}_in_{}_list'.format(feature[-5:],feature)] = df.apply(lambda x :  1 if x[feature[-5:]] in x['last_rolling_{}_cano'.format(feature)] else 0, axis =1)        

    df['conam'] = df['conam'].astype(float)

    return df

def diff_with_first_fraud_locdt(df):

    df_fraud = df[df['fraud_ind']==1].drop_duplicates(subset = ['cano'],keep = 'first')
    df_fraud = df_fraud.rename(columns = {'locdt':'first_fraud_locdt'})
    df = pd.merge(df, df_fraud[['cano','first_fraud_locdt']], how = 'left', on = ['cano'])
    df['diff_with_first_fraud_locdt'] = df['locdt'] - df['first_fraud_locdt']
    df['diff_with_first_fraud_locdt'] = np.where(df['diff_with_first_fraud_locdt']<=0, np.nan, df['diff_with_first_fraud_locdt'])

    return df
