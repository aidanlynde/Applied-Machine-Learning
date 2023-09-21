
# Question 4

# a.) In this dataset, we can use linear regression 
# to estimate the relationship between customer satisfaction (Y) and the explanatory variables (Xs), 
# which include Gender, Customer Type, Age, Type of Travel, Class, and Flight Distance. 
# However, as mentioned, some variable transformations are necessary:

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

data = pd.read_excel("Invistico_Airline.xls")

data_encoded = pd.get_dummies(data, columns=["Gender", "Customer Type", "Type of Travel", "Class"], drop_first=True)

X = data_encoded[["Age", "Flight Distance"]]
Y = data_encoded["Satisfaction"]

model = LinearRegression()
model.fit(X, Y)

# Here I used one-hot encoding to transform categorical variables... 
# into a numerical format suitable for linear regression.



# b.) For data splitting, we need to ensure a random and stratified split while considering cross-validation

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)



# c.) To perform binary classification (satisfied or not), we can use logistic regression and interpret the coefficients:

logistic_model = LogisticRegression()
logistic_model.fit(X_train, Y_train)

Y_pred = logistic_model.predict(X_test)

report = classification_report(Y_test, Y_pred)
print(report)

# Here I fit a logistic regression model and evaluated it using classification metrics like precision, recall, and F1-score.



# d.) the Probit model is a suitable choice when the dependent variable is binary. 
# It models the probability of the binary outcome (customer satisfaction) using a normal distribution. 
# To estimate the Probit model, I'll need a specialized library like statsmodels and I'll use maximum likelihood estimation (MLE)

import statsmodels.api as sm

X_train = sm.add_constant(X_train)

probit_model = sm.Probit(Y_train, X_train)
probit_results = probit_model.fit()

print(probit_results.summary())

# The Probit model is estimated using maximum likelihood estimation (MLE), 
# and the summary provides information about coefficients and their significance



# e.) To compare the two models, I will evaluate their performance on the test set using 
# classification metrics and compare coefficients:

linear_pred = model.predict(X_test)

X_test = sm.add_constant(X_test)
probit_pred = probit_results.predict(X_test)

linear_coeff = model.coef_
probit_coeff = probit_results.params

print("Linear Regression Coefficients:", linear_coeff)
print("Probit Model Coefficients:", probit_coeff)

# This calculates classification metrics and compares coefficients between 
# the linear regression and Probit models, 
# helping to determine which model performs better for predicting customer satisfaction.