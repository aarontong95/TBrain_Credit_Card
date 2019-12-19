import pandas as pd 
import numpy as np

from random import randrange

from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import average_precision_score, f1_score,recall_score, accuracy_score, roc_auc_score

from catboost import CatBoostClassifier
from model.cat_custom_eval_metric import CatCustomAveragePrecisionScore

class Cat_Model:
    
    def __init__(self, features, cat_features):    

        self.features = features
        self.cat_features = cat_features
        self.clf = None

    def build_clf(self, n_estimators = 1000, learning_rate = 0.1, num_leaves = 16, reg_alpha = 10, reg_lambda = 7, **kwargs):

        self.clf = CatBoostClassifier(  
                                        silent       = False,
                                        random_state = 10,
                                        n_estimators = 1000,
#                                         max_depth    = 8,
                                        learning_rate= 0.1,
#                                         reg_lambda   = 20,
                                        eval_metric = CatCustomAveragePrecisionScore(),
                                        od_type = 'Iter',
                                        **kwargs
                                            )     

    def run(self, data, y, groups, test, n_splits = 10, early_stopping_rounds= 100):
        
        oof_preds_LGBM = np.zeros((data.shape[0]))
        sub_preds_LGBM = np.zeros((test.shape[0]))
        df_sub_preds_LGBM = pd.DataFrame()        
        self.df_feature_importance = pd.DataFrame()
        
        if not self.clf: 
            self.build_clf()

        folds = GroupKFold(n_splits = n_splits)       
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(data, y, groups)):
            train_x, train_y = data.iloc[train_idx], y.iloc[train_idx]
            valid_x, valid_y = data.iloc[valid_idx], y.iloc[valid_idx]
            print("Starting CatGBM. Fold {},Train shape: {}, test shape: {}".format(n_fold+1, data.shape, test.shape))

            self.clf.fit(train_x, train_y,
                    eval_set = [(train_x, train_y), (valid_x, valid_y)], 
                    verbose= 10,
                    early_stopping_rounds = early_stopping_rounds,
                    cat_features = self.cat_features,
                    use_best_model=True,
                    )

            oof_preds_LGBM[valid_idx] += self.clf.predict_proba(valid_x)[:, 1]
#             sub_preds_LGBM += self.clf.predict_proba(test)[:, 1]/ (folds.n_splits)
            df_sub_preds_LGBM['fold_{}'.format(n_fold)] = self.clf.predict_proba(test)[:, 1]
            
            df_fold_importance = pd.DataFrame()
            df_fold_importance["feature"] = self.features
            df_fold_importance["importance"] = self.clf.feature_importances_
            df_fold_importance["fold"] = n_fold + 1

            self.df_feature_importance = pd.concat([self.df_feature_importance, df_fold_importance], axis=0)
            
        print('Summary:')            
        print('Cat Testing_Set average_precision_score %.6f' % average_precision_score(y, oof_preds_LGBM))

        return oof_preds_LGBM, df_sub_preds_LGBM, self.clf




