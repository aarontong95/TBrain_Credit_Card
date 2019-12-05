## Solution of TBrain's Credit Card Fraudulent Transaction Detection
The competition can be found via the link: https://tbrain.trendmicro.com.tw/Competitions/Details/10. I was ranked 3rd among 1366 teams with this solution.

## ENVIRONMENT
* python3.6

## Data Setup
* Go to https://tbrain.trendmicro.com.tw/Competitions/Details/10 
* Click the button of Download Dataset
* Download and unzip the train and test data into `data/` directory

## Install
<pre>
pip install -r requirements.txt
</pre>

## Run
The whole process can be ran in [main.ipynb](https://github.com/aarontong95/TBrain_Credit_Card/blob/master/main.ipynb).

## Model Explanation
* Total: 5 models
* Base model: Created by base features which contain 20 raw features and 13 preprocessed features.
* Special features:
4 special features to capture the information of the label, since there are some users(bacno) in both training and testing set.
* Each of the four models is trained by the base features plus one special feature.
* If the transaction in testing meets the conditions of special features, the transaction is predicted by the corresponding special model. Otherwise the transaction is predicted by the base model.
* Note: These special features may not work in real world since we don't suppose have the label.
* Features explanation [details](https://github.com/aarontong95/TBrain_Credit_Card/tree/master/preprocess)

## Key Takeaways
* Apply limited number of new features
  * In order to simplify models to have better generalizaiton
* Replace the value of categorical features of training set with NA if the value is not in testing set 
  * The model will not learn something useless when apply in testing set. The model can focus on the value which also exists in testing set. 
  * Here is the [script](https://github.com/aarontong95/TBrain_Credit_Card/blob/master/preprocess/preprocess_train_test_split.py).
* Use early stopping and split the training set by GroupKFold 
  * The model will stop training once the model performance stops improving on a hold out validation dataset. 
  * Grouping the training set by user(bacno) makes the model stop earlier which prevents overfitting. 
  * Here is the [script](https://github.com/aarontong95/TBrain_Credit_Card/blob/master/util/generate_X_y.py).
* Drop extreme cases
  * Drop the prediction of the fold if it is out of 1 standard deviation boundary, since some of the predictions of testing set are very extreme between folds. 
  * Here is the [script](https://github.com/aarontong95/TBrain_Credit_Card/blob/master/util/generate_statistic.py).
