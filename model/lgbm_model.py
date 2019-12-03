import pandas as pd 
import numpy as np

from random import randrange

from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import average_precision_score, f1_score,recall_score, accuracy_score, roc_auc_score

from lightgbm import LGBMClassifier

class LGBM_Model:
    
    def __init__(self, features):    

        self.features = features
        self.clf = None

    def build_clf(self, n_estimators = 1000, learning_rate = 0.1, num_leaves = 16, reg_alpha = 10, reg_lambda = 7):

        self.clf = LGBMClassifier(  boosting_type= 'gbdt',
                                    silent       = False,
                                    metric       = 'None',
                                    n_jobs       = -1,
                                    random_state = 10,
                                    n_estimators = n_estimators,
                                    max_depth    = -1,
                                    learning_rate= learning_rate,
                                    num_leaves   = num_leaves,
                                    reg_alpha    = reg_alpha,
                                    reg_lambda   = reg_lambda,
                                    min_child_samples = 200,
                                    # is_unbalance= 'True', 
                                    # subsample    = 1,                                    
                                    # colsample_bytree  = 1,
                                    # min_child_weight  = 1,
                                    # min_split_gain= 0.0,            
                                    # objective= 'regression_l1',
                                    # subsample_for_bin= 240000,
                                    # subsample_freq= 1,                        
                                    # class_weight= 'balanced',
                                    # scale_pos_weight = 2,
                                      )


    def run(self, data, y, groups, test , eval_metric, n_splits = 10, early_stopping_rounds= 100):
        
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
            print("Starting LightGBM. Fold {},Train shape: {}, test shape: {}".format(n_fold+1, data.shape, test.shape))

            self.clf.fit(train_x, train_y,
                    eval_set = [(train_x, train_y), (valid_x, valid_y)], 
                    eval_metric = eval_metric, 
                    verbose= 100,
                    early_stopping_rounds = early_stopping_rounds,
                    categorical_feature = 'auto',
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
        print('LGBM Testing_Set average_precision_score %.6f' % average_precision_score(y, oof_preds_LGBM))

        return oof_preds_LGBM, df_sub_preds_LGBM, self.clf

    @staticmethod 
    def lgb_f1(truth, predictions):  
        
        pred_labels = np.where(predictions>=0.275, 1, 0)
        f1 = f1_score(truth, pred_labels)
        
        return ("F1", f1, True)
    
    @staticmethod
    def lgbm_averge_precision(truth, predictions):  
        
        aps = average_precision_score(truth, predictions)
        
        return ("Averge Precision", aps, True)



