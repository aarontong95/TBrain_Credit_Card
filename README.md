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

## Model Explanation
* There are 5 models in total. 
* The base model created by the base features which contain 20 raw features and 13 preprocessed features.
* Since there are some users(bacno) in both training and testing set, I created other 4 special features captured the information of the label.
* These four features may not worked in real world since we don't suppose have the label.
* The each of other four models is trained by the base features plus one special feature.
* If the transaction in testing meets the conditions of special features, that transaction is predicted by the corresponding special model. Otherwise the transaction is predicted by the base model.
* You may take a look [here](https://github.com/aarontong95/TBrain_Credit_Card/tree/master/preprocess) for the features explanation.

## What made my model successful? 
* I just create few new features which makes my models simple and have better generalizaiton.
* I replace the value of categorical features of training set with NA if the value is not in testing set so that the model will not learn something useless when apply in testing set. The model can learn more focus on the value which exists in testing set. Here is the [script](https://github.com/aarontong95/TBrain_Credit_Card/blob/master/preprocess/preprocess_train_test_split.py).
* I split the training set by GroupKFold which i group the sample by user(bacno) so i may have better early stopping which prevents overfitting and underfitting. Here is the [script](https://github.com/aarontong95/TBrain_Credit_Card/blob/master/util/generate_X_y.py).
* Since some of the predictions of testing set are very extreme between folds, i drop the prediction of the fold if it is out of 1 standard deviation boundary. Here is the [script](https://github.com/aarontong95/TBrain_Credit_Card/blob/master/util/generate_statistic.py).
