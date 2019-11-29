def generate_statistic(df):

    df['mean'] = df.mean(axis = 1)
    df['max'] = df.max(axis = 1)
    df['min'] = df.min(axis = 1)
    df['std'] = df.std(ddof = 0, axis = 1)
    df['upper_bound_1std'] = df['mean'] + df['std']*1
    df['lower_bound_1std'] = df['mean'] - df['std']*1

    return df
