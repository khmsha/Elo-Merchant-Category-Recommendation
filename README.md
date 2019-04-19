# Elo-Merchant-Category-Recommendation
 This project is intended to help understand customer loyalty and build a recommendation engine with discount from credit card provider.

Table of Contents
=================
  * [Problem Statement](#Problem-Statement)
  * [Data](#Data)
  * [Data Wrangling](#Data-Wrangling)
  * [EDA](#EDA)
  * [Featuring Engineering](#Featuring-Engineering)
  * [Machine learning Model](#Machine-Learning-Model)
  * [Conclusion](#conclusion)


## Problem Statement
Build machine learning model to predict loyalty score for card id’s in test dataset. Training dataset contains loyalty score for each card id, historical transactions and new merchant transactions contain information about each card's transactions, and merchants.csv contains aggregate information for each merchant_id

## Data 
Data is obtained from [Kaggle](https://www.kaggle.com/c/elo-merchant-category-recommendation)

## Data Wrangling
Following data cleaning methods are used **merchant.csv**
* Missing Data 
  - Columns having inf are replaced first with NaN and then are imputed based on datatype of column as described below.
  - Columns with object datatype having NaN values are imputed with "other“
  - Columns with int and float datatype having NaN values are imputed with median
* Outliers - Outlier identification is applied for following columns. Other columns are either categorical or ID’s. 3-Sigma Rule is applied to impute outliers.
  - numerical_1
  - numerical_2
  - avg_sales_lag3
  - avg_purchases_lag3
  - avg_sales_lag6
  - avg_purchases_lag6 
  - avg_sales_lag12 
  - avg_purchases_lag12

For datasets **historical_transactions.csv** and **new_merchant_transactions.csv** – 
* Missing values (NaN)are imputed with “other” for columns with object datatype, median for columns with int and float datatype, and new category is added for columns with categorical datatype.
* Outliers  are imputed with **3-Sigma rule** for columns “purchase_amount” and “installments”
* Datetime features are created for “purchase_date”
  - Purchase year
  - Purchase month
  - Purchase day of the week
  - Purchase week of the year
  - Purchase weekend
  - Purchase hour
  - month difference - difference in numbers of months from current date to purchase date

## EDA
    **Merchant transactions Data**
1. There is strong corelation numerical_1 and numerical_2 feature.
2. There is also correlation between avg_sales and avg_purchases of 3, 6 an 12 month.
3. Merchant category ID 705 has most sales with 9% sales
4. City ID -1 has over 100000 transactions and amounts to 31% of transactions
5. Subsector ID 27 has over 50000 transactions and amounts to 15% of transactions
6. Percentage of sales in each Category
    1. 98% of the transactions does not belong to category 1
    2. 48 % of category 2 transactions are in 1.0
    3. 71 of the transactions does not belong to category 4
7. Purchase and Sales Range
    1. 53% of sales and transactions are in E range
8. December is most active sales month of the year


**Historical transactions Data**
1. There seems to be no correlation between data
2. Subsector ID 33 has over 5000000 transactions and amounts to 19% of transactions
3. City ID 33 has over 4000000 transactions and amounts to 16% of transactions
4. April Accounts for most purchase amount
5. Installments are lowest in the month of march and april. 
6. Installments are highest in the month of january and december
7. Percentage of sales in each Category
    1. 92% of the transactions does not belong to category 1
    2. 52 % of category 2 transactions are in 1.0
    3. 53 of category 3 transactions are in A
8. Subsector ID 34 has highest purchase amount
9. Subsector ID 29 has highest Installments



**New Merchant transactions Data**
1. There is a correlation between installments and purchase_amount.
2. Subsector ID 37 has over 340053 transactions and amounts to 17% of transactions
3. City ID 69 has 328916 transactions and amounts to 17% of transactions
4. Percentage of sales in each Category
    1. 97% of the transactions does not belong to category 1
    2. 54 % of category 2 transactions are in 1.0
    3. 47 of category 3 transactions are in A

5. There are no purchases in the month of march and april. Highest Purchase are in the month of May.
6. March and April has most Installments
7. Subsector ID 27 and 37 has highest Installments

**Train**
1. There is a steady increase in number of first time used cards since 2015-Jul-01.

## Featuring Engineering
**merchant.csv** – 
- One hot encoding is applied to categorical features “category_4”, “category_1”, 'category_2',                                          'most_recent_sales_range', 'most_recent_purchases_range’.
- New date Frame with categorical and anonymized measure features is created for merging historical_transactions.csv and new_merchant_transactions.csv, other features are dropped as they are only informational features about merchant ID. 
- Features considered for merging are 'merchant_id','numerical_1', 'numerical_2', 'category_2_0.0’, 'category_2_1.0', 'category_2_2.0', 'category_2_3.0', 'category_2_4.0', 'category_2_5.0', 'category_4', 'category_1’
**historical_transactions.csv and new_merchant_transactions.csv – **
- Categorical and anonymized measure features are merged with datasets historical_transactions and new_merchant_transactions
- Rows with NaN values are dropped after merging datasets as rows with NaN values are around 1%
- Category_2/category_3_purchaseAmt_mean is added by grouping category_2/category_3 and aggregating by mean over purchase_amount feature.
- One hot encoding is applied to categorical features ‘authorized_flag’, ‘category_1’, 'category_2', 'category_3’.
- Following aggregation functions is applied by grouping historical_transactions and new_merchant_transactions by card_id
'authorized_flag': ['sum', 'mean'],
  * 'category_1':['sum', 'mean'],
  * 'category_2_1.0': 'mean',
  * 'category_2_2.0': 'mean',
  * 'category_2_3.0': 'mean',
  * 'category_2_4.0': 'mean',
  * ‘category_2_5.0': 'mean’,
  * 'category_3_A': 'mean',
  * 'category_3_B': 'mean',
  * 'category_3_C': 'mean',
  * 'category_3_other': 'mean',
  * ‘state_id': 'nunique',
  * 'city_id': 'nunique',
  * 'purchase_amount': ['sum', 'mean', 'count', 'max', 'min', 'std'],
  * 'installments': ['sum', 'mean', 'max', 'min', 'std'],
  * 'purchase_date': ['min', 'max'],
  * 'month_lag': ['mean', 'max', 'min', 'std'],
  * 'card_id': ['count'],
  * 'month_diff': ['mean'],
  * 'weekend' : ['sum', 'mean'],
  * 'month': 'nunique',
  * 'hour': 'nunique’,
  * 'weekofyear': 'nunique',
  * 'dayofweek': 'nunique'
  * 'year': 'nunique',
  * 'subsector_id': 'nunique',
  * 'merchant_id': 'nunique',
  * 'merchant_category_id': 'nunique',
  * 'category_2_purchaseAmt_mean' : 'mean',
  * 'category_3_purchaseAmt_mean' : 'mean',
  * 'merchDF_numerical_1': ['mean', 'sum'],
  * 'merchDF_numerical_2': ['mean', 'sum’],
  * 'merchDF_category_2_0.0': 'mean', 
  * 'merchDF_category_2_1.0':'mean',
  * 'merchDF_category_2_2.0':'mean', 
  * 'merchDF_category_2_3.0':'mean',
  * 'merchDF_category_2_4.0':'mean', 
  * 'merchDF_category_2_5.0':'mean',
  * 'merchDF_category_4': 'mean', 
  * 'merchDF_category_1': 'mean’
- Datetime features are added to aggregated data frame
  * purchase_date_diff ---- purchase_date_max - purchase_date_min
  * purchase_date_average ----- purchase_date_diff/card_id_count
  * purchase_date_tillToday ----- Today's date - purchase_date_max
**train and test dataset**  – 
- Aggregate Data frames generated from historical_transactions and new_merchant_transactions are merged to train and test dataset
- Datetime features are added from first_active_month
  * Day of the week
  * Week of year
  * month
  * elapsed_time - Time elapsed from first active month
  * histDF_first_buy - number of days from the first buy in historical transactions dataset
  * newMerchDF_hist_first_buy - number of days from the first buy in new merchant transactions dataset
  * Convert datetime features ‘histDF_purchase_date_max’, 'histDF_purchase_date_min’, 'newMerchDF_purchase_date_max’,  'newMerchDF_purchase_date_min’ to numeric
  * card_id_total - card Id count total (count of card ID in historical_transactions and new_merchant_transactions)
  * Outlier feature is added to train dataset
  * Outlier feature is aggregated to mean by grouping on feature_1/2/3. Aggregated data frame is mapped to feature_1/2/3 in test and train

## Machine Learning Model
- Feature List is generated excluding features
  * card_id
  * first_active_month 
  * target 
  * merchant_id
  * outliers
- In this model hyperparameters are tuned using RandomizedSearchCV. Hyperparameters found in RandomizedSearchCV are used to for learning XGBClassifier.
- **Hyperparameters Tuning**
  * n_estimators - number of trees to grow. Larger the tree size better the model, but more numbers of trees can be computationally expensive and affects the performance of the model n_estimators = [4, 8, 16, 32, 64, 100, 200]
  * max_depth - depth of the tree, the more splits it has and it captures more information about the data. But as the tree gets very deep, it might lead to overfitting max_depth = [4, 8, 10, 12, 16, 32, 64]
  * min_child_weight - Minimum sum of instance weight needed in a child. min_child_weight = [2, 4, 6, 8, 10, 12, 16, 32, 64]
  * gamma - [0.1, 0.2, 0.3, 0.4, 0.5]
  * colsample_bytree - Subsample ratio of columns when constructing each tree. colsample_bytree = [0.2, 0.4, 0.6, 0.8]
  * colsample_bylevel - Subsample ratio of columns for each split, in each level colsample_bylevel = [0.2, 0.4, 0.6, 0.8]
- **Tuned Hyperparameters** are n_estimators - 100, max_depth - 8, min_child_weight - 32, gamma – 0.2, colsample_bytree- 0.2, colsample_bylevel – 0.6
- RMSE is calculated on target and values predicted from train dataset, which is 3.38569

## Conclusion
- Top five features impacting model impacting loyalty score
  * histDF_year_nunique -- number of unique year in a card ID transactions in Historical transactions dataset
  * histDF_month_nunique -- number of unique months in a card ID transactions in Historical transactions dataset
  * newMerchDF_purchase_days_tillToday -- number of purchase days from last purchase date in new merchant transactions dataset
  * histDF_purchase_date_max -- Most recent purchase date of card ID in Historical transactions dataset
  * newMerchDF_month_nunique -- number of unique months in a card ID transactions in new merchant transactions dataset

**Recommendation** -
- If the loyalty score of a card is low, then discount in top important category can sent to card holder.
- Loyalty score can be monitored monthly and if the loyalty score decrease then a discount in most important category can set to card holder.

