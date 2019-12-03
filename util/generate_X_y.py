import numpy as np 

def generate_X_y(df_train, df_test, label, features): 
    
    # Create bacno_transfer for splitting the training set by GroupKFold
    df_train['bacno_transfer'] = np.where(df_train['bacno'].astype(int) >= 0, -9999, df_train['bacno_original'])
        
    return df_train[features], df_train[label], df_train['bacno_transfer'], df_test[features]
