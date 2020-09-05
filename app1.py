# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 12:22:41 2020

@author: imarv
"""


# load libraries
import numpy as np
import pandas as pd
import pickle
import PIL
import streamlit as st

#import the data
data = pd.read_csv("bank_churn_train.csv")

with open("best_xgb.pickle", "rb") as f:
    ohe = pickle.load(f)
    sc = pickle.load(f)
    clf_xgb = pickle.load(f)
    

def data_preparation(balance,salary,credit,age,IsActive,NumOfProducts,tenure,gender,geography):
    mean_age = 38.90725
    mean_credit = 649.4
    mode_products = 1
    mean_salary = 100070.5
    balance_salary_ratio = balance/salary
    balance_or_not = 1 if balance > 0.0 else 0
    creditscore_age_ratio = credit/age
    creditscore_age_ratio_log = np.log10(creditscore_age_ratio)
    Better_Age_Credit = 1 if ((age < mean_age) and (credit > mean_credit)) else 0 
    Better_Age_Credit_Active = 1 if (Better_Age_Credit == 1 and IsActive == 1) else 0
    multi_products = 1 if NumOfProducts > 1 else 0
    Valuable_customer = 1 if ((Better_Age_Credit_Active == 1) and (NumOfProducts > mode_products)) else 0 
    tenure_age_ratio = tenure/age
    high_salary_age = 1 if (salary > mean_salary and age < mean_age) else 0
    
    # encoding gender
    gender = 1 if (gender.lower() == 'male') else 0
    
    # one hot encoding country
    if geography.lower().startswith('f'):
        geography = 'France'
    elif geography.lower().startswith('g'):
        geography = 'Germany'
    else:
        geography = 'Spain'
    geo_ohe = ohe.transform(np.array(geography).reshape(1,-1))[0]
    
    # Standardization
    test_std = sc.transform(np.array([credit,age,tenure,balance,NumOfProducts,salary,balance_salary_ratio,creditscore_age_ratio,creditscore_age_ratio_log,tenure_age_ratio]).reshape(1, -1))[0]
    
    credit_std,age_std,tenure_std,balance_std,NumOfProducts_std = test_std[0],test_std[1],test_std[2],test_std[3],test_std[4]
    salary_std,balance_salary_ratio_std,creditscore_age_ratio_std = test_std[5],test_std[6],test_std[7]
    creditscore_age_ratio_log_std,tenure_age_ratio_std = test_std[8],test_std[9]
    
    test = [credit_std, gender, age_std, tenure_std, NumOfProducts_std, IsActive, salary_std, balance_or_not,
            creditscore_age_ratio_std,creditscore_age_ratio_log_std,Better_Age_Credit,Better_Age_Credit_Active,
            multi_products,Valuable_customer, tenure_age_ratio_std,high_salary_age, geo_ohe[0], geo_ohe[1], geo_ohe[2]]
    
    return np.array(test).reshape(1, -1)

#lr_imp.predict_proba(np.array(test).reshape(1, -1))[:,1]
    
def predict_bank_churn(balance,salary,credit,age,IsActive,NumOfProducts,tenure,gender,geography):
    
    test = data_preparation(balance,salary,credit,age,IsActive,NumOfProducts,tenure,gender,geography)
    y_pred_proba = clf_xgb.predict_proba(test)[:,1]
    print(y_pred_proba)
    return y_pred_proba    
    
def main():
    st.title("Bank Churn Predictor")
    st.subheader("Created by: Aravind R")
    st.sidebar.title("Why customer retention is important?")
    st.sidebar.markdown("""1. Save Money On Marketing
2. Repeat Purchases From Repeat Customers Means Repeat Profit
3. Free Word-Of-Mouth Advertising
4. Retained Customers Will Provide Valuable Feedback
5. Previous Customers Will Pay Premium Prices. 

Why and when will a customer leave his/her bank could be a challenging question to answer.

Here, I have taken a data from kaggle where all the historical information about a customer and whether he/she left the bank or not is available.

The goal is to use the power of data science to help the bank identify those who are likely to leave the bank in future.""")
    #checking the data
    st.write("Will the customer stay in your bank?")
    check_data = st.checkbox("Data sample")
    if check_data:
        st.write(data.head(10))
    st.write("Using Machine Learning let's try to predict Churn")
    st.write("Slide/Select Input values:")
    
    #input the numbers
    balance = st.slider("Customer's account balance",int(data.Balance.min()),int(data.Balance.max()),int(data.Balance.mean()) )
    salary  = st.slider("Customer's estimated Salary?",int(data.EstimatedSalary.min()),int(data.EstimatedSalary.max()),int(data.EstimatedSalary.mean()))
    credit  = st.slider("Customer's credit score?",int(data.CreditScore.min()),int(data.CreditScore.max()),int(data.CreditScore.mean()) )
    age     = st.slider("Customer's age?",int(data.Age.min()),int(data.Age.max()),int(data.Age.mean()))
    IsActive= st.radio("Active Member?",data.IsActiveMember.unique())
    NumOfProducts = st.selectbox("Number of Products?",list(data.NumOfProducts.unique()),0)
    #NumOfProducts = st.slider("Number of Products?",int(data.NumOfProducts.min()),int(data.NumOfProducts.max()),int(data.NumOfProducts.mode()))
    tenure     = st.slider("Tenure?",int(data.Tenure.min()),int(data.Tenure.max()),int(data.Tenure.mode()))
    gender     = st.radio("Gender", data.Gender.unique())
    geography  = st.selectbox("Customer's Geography?", list(data.Geography.unique()),0)
    
    prediction_proba = predict_bank_churn(balance,salary,credit,age,IsActive,NumOfProducts,tenure,gender,geography)
    prediction = 'Yes' if prediction_proba > 0.48 else 'No'
    #checking prediction house price
    if st.button("Predict!"):
        st.header("Will the customer leave in the near future: {}".format(prediction))
        st.subheader("Customer probability to leave: {}".format(str(int(prediction_proba[0]*100))+'%'))

if __name__=='__main__':
    main()    