import pandas as pd
import numpy as np


def preprocess_init(df_train, df_test, bool_features):

    df = pd.concat([df_train, df_test], sort=True)

    df['days'] = np.select([df['locdt']<=30, [(df['locdt']>30) & (df['locdt']<=60)] , [(df['locdt']>60) & (df['locdt']<=90)], [(df['locdt']>90) & (df['locdt']<=120)]],[30,60,90,120])[0]

    df = preprocess_bool(df, bool_features)

    df = df.sort_values(by = ['bacno','cano','locdt','loctm']).reset_index(drop = True)

    return df
                
def preprocess_bool(df, bool_features):
    
    for feature in bool_features:
        df[feature] = np.select([df[feature]=='Y',df[feature]=='N'],[1,0]) 
    
    return df
