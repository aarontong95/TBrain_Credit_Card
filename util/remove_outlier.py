def remove_outlier(df):

    proba_list = []
    for i in range(0,10):
        if df[i] < df['upper_bound'] and df[i] > df['low_bound']:
            proba_list.append(df[i])
    mean = sum(proba_list)/len(proba_list)
    
    return mean