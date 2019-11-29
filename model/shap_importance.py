import numpy as np
import pandas as pd
import shap


def shap_importance(clf, X):
    
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)
    shap_values = np.absolute(shap_values)
    shap_values = shap_values.sum(axis=0)

    df_shap = pd.DataFrame()
    df_shap["feature"] = X.columns
    df_shap["importance"] = shap_values
    df_shap = df_shap.sort_values(['importance'],ascending=False).reset_index(drop =True)
    
    return shap_df
