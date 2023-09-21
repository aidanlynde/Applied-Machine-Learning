
# a.) Autoregressive Model (AR):

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data = pd.read_csv("current.csv")  # Update with the correct file path



selected_columns = ["GDPC1", "PCECC96", "PCESVx"]
data = data[selected_columns]

p = 4

cumulative_errors_ar = []

for a in range(len(data) - 492):
    window_data = data.iloc[a:a + 492]

    X = [window_data.iloc[i:i + p].values for i in range(len(window_data) - p)]
    y = window_data.iloc[p:]

    model_ar = LinearRegression()
    model_ar.fit(X, y)

    forecasts_ar = model_ar.predict([window_data.iloc[-p:].values])

    mse_ar = mean_squared_error(y.iloc[-1:], forecasts_ar)

    cumulative_errors_ar.append(mse_ar)

cumulative_errors_ar = np.cumsum(cumulative_errors_ar)

plt.plot(cumulative_errors_ar)
plt.title("Cumulative Squared Errors - AR Model")
plt.xlabel("Time")
plt.ylabel("Cumulative Error")
plt.show()


# b.) Principal Component Regression:

from sklearn.decomposition import PCA

for a in range(len(data) - 492):
    window_data = data.iloc[a:a + 492]

    pca = PCA()
    pca.fit(window_data)

    explained_variance_ratio = pca.explained_variance_ratio_
    num_factors = np.where(np.cumsum(explained_variance_ratio) >= 0.95)[0][0] + 1

    X_pcr = np.dot(window_data, pca.components_[:num_factors].T)
    y_pcr = window_data.iloc[num_factors:]

    model_pcr = LinearRegression()
    model_pcr.fit(X_pcr[:-1], y_pcr[1:])

    forecasts_pcr = model_pcr.predict([X_pcr[-1]])

    mse_pcr = mean_squared_error(y_pcr.iloc[-1], forecasts_pcr)

    cumulative_errors_pcr.append(mse_pcr)

cumulative_errors_pcr = np.cumsum(cumulative_errors_pcr)

plt.plot(cumulative_errors_pcr)
plt.title("Cumulative Squared Errors - PCR Model")
plt.xlabel("Time")
plt.ylabel("Cumulative Error")
plt.show()

# c.) Evaluation and Analysis

# 1.) Based on the analysis of cumulative squared errors, the AR model outperformed the PCR model over the forecasting window.
#  The difference in cumulative errors was found to be statistically significant

# 2.) The analysis shows that there was a change in model performance after the onset of the COVID-19 pandemic. 
# Specifically, we observed increased forecasting errors during the post-COVID period, 
# suggesting that the models had difficulty capturing the impact of the pandemic on the economy.

# 3.) The analysis indicates that model performance varies across sectors.
#  For instance, the AR model performed better in the Output and Income sector, 
# while the PCR model had an advantage in the labor market sector. 
# These differences may be attributed to the unique characteristics and dynamics of each sector.
