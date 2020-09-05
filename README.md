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

    7. Age: Age of the customer. Type : int64
    8. Tenure: Duration for which the loan amount is sanctioned.Assuming it to be in years Type : int64
    9. Balance: The amount of money the customer has available in his account. Type: int64
    10. NumOfProducts: How many accounts, bank account affiliated products the person has. Type: int64
    11. HasCrCard: whether the person holds a credit card or not. 1 means he/she has a credit card and 0 means he/she doesn't. Type: int64
    12. IsActiveMember: Whether the customer is actively using the account. However, the values are subjective. Type: int64
    13. EstimatedSalary: The person's approximate salary. Type: float64
    14. Exited: Whether the customer has left the bank or not. 1 means he/she left and 0 means he/she didn't. Type: int64

From the above, we will not require RowNumber, CustomerId, and Surname are related to individuals.


## Memory Handling
Memory usage in python is a key task. In case of huge datasets memory handling is not easy. It is always a good practice to reduce memory of the data.

```python
# Before Memory reduction
print("Total memory used before Memory reduction {:5.2f}Mb".format(bank_data.memory_usage().sum() / 1024**2))
```

Total memory used before Memory reduction  0.84Mb
    
```python
# After Memory reduction
bank_data = reduce_memory(bank_data)
print("Total memory used after Memory reduction {:5.2f}Mb".format(bank_data.memory_usage().sum() / 1024**2))
```

Memory usage decreased to  0.31 Mb (63.6% reduction)
Total memory used after Memory reduction  0.31Mb
    

## Exploratory Data Analysis (EDA)
The purpose of EDA is to understand how different variables are related to our target (Exited) variable.

```python
import plotly.graph_objects as go

labels = ['Exited','Continued']
values =  [bank_data.Exited[bank_data['Exited']==1].count(), bank_data.Exited[bank_data['Exited']==0].count()]
colors = ['red', 'darkorange']
fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
fig.update_traces(hole=.4, hoverinfo='label+value',  textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))
fig.update_layout(
    title_text="Ratio of Customer Churned and Retained")
fig.show()
fig.write_html(fig, file='pie_chart.html', auto_open=True)

```
![](https://github.com/imAravindR/imAravindR.github.io/tree/master/images/perceptron/pie_chart.png)
