# Bank_Churn
Analysing Data, Building ML Models and Deploying
The app is hoted on https://bank-churn-streamlit-app.herokuapp.com/

Why Customer retention is important? [source](https://www.dcrstrategies.com/customer-incentives/5-reasons-customer-retention-business/)

    1. Save Money On Marketing
    2. Repeat Purchases From Repeat Customers Means Repeat Profit
    3. Free Word-Of-Mouth Advertising
    4. Retained Customers Will Provide Valuable Feedback
    5. Previous Customers Will Pay Premium Prices. 

Why and when will a customer leave his/her bank could be a challenging question to answer.

Here, we have a data from kaggle where all the historical information about a customer and whether he/she left the bank or not is available.

Our goal is to use the power of data science to help the bank identify those who are likely to leave the bank in future.

Refer: https://imaravindr.github.io/perceptron/

# Usage

For Exploratory Data Analysis (EDA) and Machine Learning Model to predict churn refer [Bank_Churn.ipynb](https://github.com/imAravindR/Bank_Churn/blob/master/Bank_Churn.ipynb)

To run the app in local:
  1. Download the repository
  2. Create a virtual environment (For anaconda user [refer](https://www.youtube.com/watch?v=ntxwMtFnW94))
  3. Setup requirements.txt (for windows users run pip install -r requirements.txt)
  4. After install the required packages. Open anaconda promt (anaconda cmd/Terminal) and type cd 'directory of repository in local machine'
  5. Activate conda environment. (conda activate env_name)
  6. Run app.py (streamlit run app.py)
  
     Example: (base) C:\Users\aravind>cd C:\Users\aravind\Desktop\Telecom_churn
              (base) C:\Users\aravind\Desktop\Telecom_churn>conda activate churn_env
              (bank_env) C:\Users\aravind\Desktop\Telecom_churn>streamlit run app.py

# App Demo
![](https://github.com/imAravindR/Bank_Churn/blob/master/streamlit-app-2020-06-22-18-06-58.gif)

