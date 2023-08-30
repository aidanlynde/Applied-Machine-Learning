import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
num_repetitions = 1000
num_observations = 501

# Initialize arrays to store prediction errors
mse_mean = np.zeros(num_repetitions)
mse_median = np.zeros(num_repetitions)

# Monte Carlo Simulation
for _ in range(num_repetitions):
    # Generate random data
    normal_data = np.random.normal(0, 1, num_observations)
    t_data = np.random.standard_t(4, num_observations)
    chi2_data = np.random.chisquare(4, num_observations)
    
    # Calculate mean and median excluding the last observation
    mean_normal = np.mean(normal_data[:-1])
    mean_t = np.mean(t_data[:-1])
    mean_chi2 = np.mean(chi2_data[:-1])
    
    median_normal = np.median(normal_data[:-1])
    median_t = np.median(t_data[:-1])
    median_chi2 = np.median(chi2_data[:-1])
    
    # Predict the last observation using the estimated mean and median
    pred_normal_mean = mean_normal
    pred_t_mean = mean_t
    pred_chi2_mean = mean_chi2
    
    pred_normal_median = median_normal
    pred_t_median = median_t
    pred_chi2_median = median_chi2
    
    # Calculate prediction error
    mse_mean[_] = (normal_data[-1] - pred_normal_mean)**2
    mse_mean[_] += (t_data[-1] - pred_t_mean)**2
    mse_mean[_] += (chi2_data[-1] - pred_chi2_mean)**2
    
    mse_median[_] = (normal_data[-1] - pred_normal_median)**2
    mse_median[_] += (t_data[-1] - pred_t_median)**2
    mse_median[_] += (chi2_data[-1] - pred_chi2_median)**2

# Compute mean squared errors
mse_mean_avg = np.mean(mse_mean)
mse_median_avg = np.mean(mse_median)

print("Mean Squared Error (Using Mean Estimation):", mse_mean_avg)
print("Mean Squared Error (Using Median Estimation):", mse_median_avg)