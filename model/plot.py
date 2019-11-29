import matplotlib.pyplot as plt
import seaborn as sns


def plot_importance(df_feature_importance, save_path = None):
        
    cols = df_feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(by = "importance", ascending = False).index
    best_features = df_feature_importance.loc[df_feature_importance.feature.isin(cols)]
    plt.figure(figsize=(10, 16))
    sns.barplot(x = "importance", y = "feature", data = best_features.sort_values(by = "importance", ascending = False))
    plt.title('LGBM Features')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)                

    plt.show()