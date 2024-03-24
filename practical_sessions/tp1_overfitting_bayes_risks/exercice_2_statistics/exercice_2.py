"""
Study overfitting and variance of the test error estimation
by monitoring the R2 train and test scores after subsampling the datasets
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression

def main():
    X_train = np.load(os.path.join("data", "X_train.npy"))
    X_test = np.load(os.path.join("data", "X_test.npy"))
    y_train = np.load(os.path.join("data", "y_train.npy"))
    y_test = np.load(os.path.join("data", "y_test.npy"))

    # rng = np.random.default_rng()

    """
    Study the variance of the test error estimation
    by subsampling the test set
    n_train will also have an influence on the result
    """

    # idx = np.random.choice(np.arange(len(X_train)), 300, replace=False)
    # reg = LinearRegression().fit(X_train[idx], y_train[idx])
  
    # std = []
    # n_test_list = np.arange(10, len(X_test), 10)
    # for i in n_test_list:
    #     scores = []
    #     for _ in range(150):
    #         idx_2 = np.random.choice(np.arange(len(X_train)), i, replace=False)
    #         scores.append(reg.score(X_test[idx_2], y_test[idx_2]))
    #     std.append(np.array(scores).std())
        
    # plt.plot(n_test_list, std, "o", alpha=0.7)
    # plt.show()

    """
    Study overfitting
    by subsampling the train set
    """

    n_test_list = np.arange(10, 500, 10)
    scores = []
    for i in n_test_list:
        idx = np.random.choice(np.arange(len(X_train)), i, replace=False)
        reg = LinearRegression().fit(X_train[idx], y_train[idx])
        scores.append(reg.score(X_train[idx], y_train[idx]) - reg.score(X_test, y_test))
    plt.plot(n_test_list, scores, "o", alpha=0.7)
    plt.yscale("log")
    plt.show()

if __name__ == "__main__":
    main()
