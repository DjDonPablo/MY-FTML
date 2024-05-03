"""
Template of functions to edit to fulfill the exercises.
If you prefer, you can also start from scratch and write your own functions.
"""

import os
import numpy as np
from constants import SIGMA

def generate_output_data(
    X: np.ndarray, theta_star: np.ndarray, sigma: float, rng, n_tests: int
) -> np.ndarray:
    n = X.shape[0]
    noise = rng.normal(0, sigma, size=(n, n_tests))
    y = X @ theta_star + noise
    return y

def OLS_estimator(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    covariance_matrix = X.T @ X
    inverse_covariance = np.linalg.inv(covariance_matrix)
    theta_hat = inverse_covariance @ (X.T @ y)
    return theta_hat

def ols_risk(n, d, n_tests) -> tuple[float, float]:
    return 1, 1
    # instantiate a PRNG
    rng = np.random.default_rng()

    data_path = os.path.join("data", f"design_matrix_n={n}_d={d}.npy")
    X = np.load(data_path)

    # Bayes predictor
    theta_star = rng.uniform(0, 1, size=(d, 1))

    # run several simulations to have an estimation of the excess risk
    y = generate_output_data(X, theta_star, SIGMA, rng, n_tests)

    # compute the OLS estimator
    theta_hat = OLS_estimator(X, y)

    distances = np.linalg.norm(theta_hat - theta_star, axis=0)
    relative_distances = distances / np.linalg.norm(theta_star)
    std_relative_distance = relative_distances.std()

    # generate test data
    y_test = generate_output_data(X, theta_star, SIGMA, rng, n_tests)

    # compute predictions of each OLS estimator
    y_pred = X @ theta_hat

    mean_test_error = np.linalg.norm(y_pred - y_test) ** 2 / (n * n_tests)

    return mean_test_error, std_relative_distance

def ridge_estimator(X: np.ndarray, y: np.ndarray, lambda_: int, n: int, d: int) -> np.ndarray:
    return (1/n) * np.linalg.inv(X.T @ X / n + lambda_ * np.identity(d)) @ X.T @ y
    

def ridge_risk(n, d, lambda_, n_tests) -> tuple[float, float]:
    # instantiate a PRNG
    rng = np.random.default_rng()

    data_path = os.path.join("data", f"design_matrix_n={n}_d={d}.npy")
    X = np.load(data_path)

    # Bayes predictor
    theta_star = rng.uniform(0, 1, size=(d, 1))

    # run several simulations to have an estimation of the excess risk
    y = generate_output_data(X, theta_star, SIGMA, rng, n_tests)

    theta_hat = ridge_estimator(X, y, lambda_, n, d)

    distances = np.linalg.norm(theta_hat - theta_star, axis=0)
    relative_distances = distances / np.linalg.norm(theta_star)
    std_relative_distance = relative_distances.std()

    # generate test data
    y_test = generate_output_data(X, theta_star, SIGMA, rng, n_tests)

    # compute predictions of each OLS estimator
    y_pred = X @ theta_hat

    mean_test_error = np.linalg.norm(y_pred - y_test) ** 2 / (n * n_tests)

    return mean_test_error, std_relative_distance