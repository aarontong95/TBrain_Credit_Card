def remove_outlier(df):

    proba_list = []
    for i in range(0,10):
        if df[i] < df['upper_bound_1std'] and df[i] > df['lower_bound_1std']:
            proba_list.append(df[i])
    mean = sum(proba_list)/len(proba_list)
    
    return mean