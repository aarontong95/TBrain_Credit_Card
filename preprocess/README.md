# Feature Explanation
* For the explanation of raw features, please take a look [here](https://github.com/aarontong95/TBrain_Credit_Card/blob/master/data/dataset_description.pdf).
* For the explanation in chinese, please take a look in [Features_Explanation_Chinese.pdf](https://github.com/aarontong95/TBrain_Credit_Card/blob/master/preprocess/Features_Explanation_Chinese.pdf).
## Transaction Frequency Features 
* cano_days_txkey_count 
  * The number of times of transaction during the period(30 days)
* cano_locdt_txkey_count
  * The number of times of transaction during the day(locdt)
* bacno_locdt_mchno_txkey_count
  * The number of times of transaction for the merchant(mchno) during the period(30 days)
* Here is the [script](https://github.com/aarontong95/TBrain_Credit_Card/blob/master/preprocess/preprocess_transaction_frequecy.py).                                    

## Time Features 
* last_time_days
  * The time(loctm) difference between the transaction and previous transaction
* next_time_days
  * The time(loctm) difference between the transaction and next transaction
* cano_locdt_global_time_std 
  * The standard deviation of the transaction time(loctm) during the day(locdt)             
* Here is the [script](https://github.com/aarontong95/TBrain_Credit_Card/blob/master/preprocess/preprocess_time.py).                                    

## Changing Card Features
* diff_locdt_with_last_trans_cano
  * The days(locdt) difference between the transaction and the last transaction with same card(cano)
* diff_locdt_of_two_card
  * The days(locdt) difference between cardA(canoA) and cardB(canoB) with same user(bacno)
* Here is the [script](https://github.com/aarontong95/TBrain_Credit_Card/blob/master/preprocess/preprocess_change_card.py).                                    

## Transaction Amount(conam) Features 
* cano_locdt_conam_min
  * The minimum amount(conam) of transaction of the card(cano) during the day(locdt)
* cano_locdt_conam_max
  * The maximum amount(conam) of transaction of the card(cano) during the day(locdt)
* diff_gtime_with_conam_zero_trans_locdt
  * The time(loctm) difference between the transaction and the transaction with zero amount(conam)(if exist) 
* Here is the [script](https://github.com/aarontong95/TBrain_Credit_Card/blob/master/preprocess/preprocess_conam.py).                                    

## Merchant(mchno) Features
* bacno_mchno_locdt_head_tail_diff
  * The days(locdt) difference between the first and last transaction with same card(cano) and merchant(mchno)
* cano_days_mchno_index
  * The n-th transaction with same card(cano) and merchant(mchno)
* Here is the [script](https://github.com/aarontong95/TBrain_Credit_Card/blob/master/preprocess/preprocess_mchno.py).                                    

## Special Features
* mchno_in_normal_mchno_list
  * Whitelist of merchant(mchno) with same user(bacno)
* mchno_in_fraud_mchno_list
  * Blacklist of merchant(mchno) with same card(cano)
* conam_in_fraud_conam_list
  * Blacklist of transaction amount(cano) with same card(cano)
* diff_with_first_fraud_locdt
  * The days(locdt) difference between the transaction and the first fraudulent transaction(if exist) 
* Here is the [script](https://github.com/aarontong95/TBrain_Credit_Card/blob/master/preprocess/preprocess_special_features.py).                                    
