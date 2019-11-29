import pandas as pd
import numpy as np


def preprocess_change_card(df):

    df_tem = df.groupby(['bacno','cano','days']).agg(['max','min'])['locdt'].reset_index().sort_values(by = ['cano','max'])
    df_tem['next_card_min'] = df_tem.groupby(['bacno','days'])['min'].shift(-1)
    df_tem['next_card_min'] = np.where(df_tem['max'] - df_tem['next_card_min']>=0, np.nan, df_tem['next_card_min'])
    df_tem['diff_locdt_of_two_card'] = df_tem['max'] - df_tem['next_card_min']
    df_tem = df_tem.rename(columns = {'max':'cano_last_trans_locdt'})
    df_tem = df_tem.iloc[:,list(range(1,7))]
    df = pd.merge(df,df_tem,how = 'left', on = ['cano','days'])
    df['diff_locdt_with_last_trans_cano'] = df['locdt'] - df['cano_last_trans_locdt']
    
    df['diff_locdt_with_last_trans_days_cano'] = df['days'] - df['cano_last_trans_locdt']

    return df

