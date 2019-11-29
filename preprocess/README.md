# Feature Explanation
For the raw features explanation, please take a look [here](https://github.com/aarontong95/TBrain_Credit_Card/blob/master/data/dataset_description.pdf)
## Transaction Frequency Feautres 
* cano_days_txkey_count 
  * The number of times of transaction during the period(30 days)
* cano_locdt_txkey_count
  * The number of times of transaction during the day
* bacno_locdt_mchno_txkey_count
  * The number of times of transaction for specific merchant(mchno) during the period(30 days)
                                    

## Time Feautres 
* last_time_days
  * The time difference between the transaction and last transaction
* next_time_days
  * The time difference between the transaction and next transaction
* cano_locdt_global_time_std 
  * The standard deviation of the transaction time during the day             

## Change Card Feautres
* diff_locdt_with_last_trans_cano
* diff_locdt_of_two_card


## Transaction Amount(conam) Feautres 
* cano_locdt_conam_min
* cano_locdt_conam_max
* diff_gtime_with_conam_zero_trans_locdt


## Merchant(mchno) Features
* bacno_mchno_locdt_head_tail_diff
* cano_days_mchno_index


## Special Feautures
* mchno_in_normal_mchno_list
* mchno_in_fraud_mchno_list
* conam_in_fraud_conam_list
* diff_with_first_fraud_locdt
