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
    Add Exploratory data analysis

## Featuring Engineering
    Add feature enginerring methods

## Machine Learning Model
    Add machine learning model

## Conclusion
    Add conclusion
