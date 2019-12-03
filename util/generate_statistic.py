def generate_statistic(df):
    
    df_10_folds = df.copy()
    df['max'] = df_10_folds.max(axis = 1)
    df['min'] = df_10_folds.min(axis = 1)
    
    # Create lower and upper bound with 1 standard deviation 
    # for dropping the fold of prediction if it is out of boundary
    df['std'] = df_10_folds.std(ddof = 0, axis = 1)
    df['mean'] = df_10_folds.mean(axis = 1)
    df['upper_bound_1std'] = df['mean'] + df['std']*1
    df['lower_bound_1std'] = df['mean'] - df['std']*1

    return df
