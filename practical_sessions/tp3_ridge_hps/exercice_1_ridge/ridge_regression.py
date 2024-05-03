from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
import numpy as np
import os


def OLS_estimator(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    covariance_matrix = X.T @ X
    inverse_covariance = np.linalg.inv(covariance_matrix)
    theta_hat = inverse_covariance @ (X.T @ y)
    return theta_hat

def ridge_risk(n, d, lambda_, n_tests) -> tuple[float, float]:
    """
    Statistical evaluation of the excess risk of the Ridge regression
    estimator

    n_test times, do:
        - Draw output vector Y, according to the linear model, fixed
        design setup.
        - compute the corresponding Ridge estimator
        - generate a test set in order to have an estimation of the excess risk of
        this estimator (generalization error)  

    Parameters:
        n (int): number of samples in the dataset
        d (int): dimension of each sample (number of features)
        n_tests (int): number of simulations run

    Returns:
        risk_estimation (float): estimation of the excess risk of the OLS
        estimator in this setup.
    """
    # instantiate a PRNG
    rng = np.random.default_rng()

    # use a specific design matrix
    # data_path = os.path.join("data", f"design_matrix_n={n}_d={d}.npy")
    # if not os.path.exists(data_path):
    #     print("generate matrix")
    #     X = generate_low_rank_design_matrix(n, d, rng)
    # else:
    #     X = np.load(data_path)

    return 1

def get_X(n, d) -> np.ndarray:
    data_path = os.path.join("data", f"design_matrix_n={n}_d={d}.npy")
    return np.load(data_path)

def train_ridge(X, alpha, n_tests) -> float:
    reg = Ridge(alpha=alpha)
    reg.fit(X) 
    return 1 
    
def train_ols(X, n_tests) -> float:
    reg = LinearRegression()
    reg.fit(X)
    return 1
    
def main() -> None:
    n = 30
    d_list = [10, 20, 30, 40] 
    
    exponents = [k for k in range(-6, 6)]
    lambda_list = [10 ** (u) for u in exponents]

    n_tests = int(1e4)
   
    ols_risk = [] 
    ridge_risk = []
    for d in d_list:
        X = get_X(n, d)
        ols_risk.append(train_ols(X, n_tests))
        tmp_ridge = []
        for l in lambda_list:
            tmp_ridge.append(train_ridge(X, l, n_tests))
        ridge_risk.append(tmp_ridge)
    
    
if __name__ == "__main__":
    main()