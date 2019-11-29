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
* Since there are some users(cano) in both training and testing set, I created other 4 special features using the information of the label.
* These four features may not worked in real world since we don't suppose have the label.
* The each of other four models is trained by the base features plus one special feature.
* If the transaction in testing meets the conditions of special features, that transaction is predicted by the corresponding special model. Otherwise the transaction is predicted by the base model.
* You may take a look [here](https://github.com/aarontong95/TBrain_Credit_Card/tree/master/preprocess) for the features explanation.
