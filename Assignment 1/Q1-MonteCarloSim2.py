import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
num_repetitions = 1000
num_observations = 500
true_beta = 2.5

# Initialize arrays to store estimators and t-statistics
estimators = np.zeros(num_repetitions)
t_statistics = np.zeros(num_repetitions)

# Monte Carlo Simulation
for i in range(num_repetitions):
    # Generate random data
    U = np.random.normal(0, 1, num_observations)
    X = np.random.normal(0, np.sqrt(2), num_observations)
    Y = true_beta * X + U
    
    # OLS estimation
    X_matrix = np.column_stack((np.ones(num_observations), X))
    beta_hat = np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T @ Y
    estimators[i] = beta_hat[1]  # Save the estimator
    
    # Calculate t-statistic
    se_beta = np.sqrt(np.sum((Y - X_matrix @ beta_hat)**2) / (num_observations - 2))
    t_statistic = (beta_hat[1] - true_beta) / se_beta
    t_statistics[i] = t_statistic  # Save the t-statistic

# Plot histograms
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(estimators, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(x=true_beta, color='red', linestyle='dashed', linewidth=2, label='True Beta')
plt.xlabel('Estimator')
plt.ylabel('Frequency')
plt.title('Histogram of Estimators for Beta')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(t_statistics, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(x=0, color='red', linestyle='dashed', linewidth=2, label='Null Hypothesis')
plt.xlabel('T-Statistic')
plt.ylabel('Frequency')
plt.title('Histogram of T-Statistics')
plt.legend()

plt.tight_layout()
plt.show()
