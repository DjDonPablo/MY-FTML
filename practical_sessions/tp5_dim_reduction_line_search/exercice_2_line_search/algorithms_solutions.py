import numpy as np


def gamma_line_search(H: np.ndarray, gradient: np.ndarray) -> float:
    """
    Exact line search gamma
    """
    square_norm = np.linalg.norm(gradient) ** 2
    d = gradient.shape[0]
    Hgrad = (H @ gradient).reshape(d)
    grad_reshape = gradient.reshape(d)
    inner_product = np.dot(Hgrad, grad_reshape)
    return square_norm / inner_product


def line_search(
    X,
    H,
    y,
    theta_star,
    n_iterations,
    ax_linear,
    ax_log,
):
    """
    Perform exact line search gradient descent
    """
    n, d = X.shape
    theta_0 = np.zeros((d, 1))
    LS_squared_distances_to_opt = list()
    theta_LS = theta_0.copy()

    for _ in range(1, n_iterations + 1):
        LS_squared_distances_to_opt.append(np.linalg.norm(theta_LS - theta_star) ** 2)
        grad = gradient(theta_LS, H, X, y)
        gamma_star = gamma_line_search(H, grad)
        theta_LS -= gamma_star * gradient(theta_LS, H, X, y)

    label = "line search"
    x_plot = range(1, n_iterations + 1)
    ax_linear.plot(x_plot, LS_squared_distances_to_opt, label=label)
    ax_log.plot(x_plot, LS_squared_distances_to_opt, label=label)


def gradient_descent(
    X,
    H,
    y,
    theta_star,
    gamma,
    n_iterations,
    ax_linear,
    ax_log,
):
    """
    Perform vanilla gradient descent
    """
    n, d = X.shape
    theta_0 = np.zeros((d, 1))
    GD_squared_distances_to_opt = list()
    theta_GD = theta_0.copy()
    for _ in range(1, n_iterations + 1):
        GD_squared_distances_to_opt.append(np.linalg.norm(theta_GD - theta_star) ** 2)
        theta_GD -= gamma * gradient(theta_GD, H, X, y)

    label = r"$\gamma=$" f"{gamma}"
    x_plot = range(1, n_iterations + 1)
    ax_linear.plot(x_plot, GD_squared_distances_to_opt, label=label)
    ax_log.plot(x_plot, GD_squared_distances_to_opt, label=label)


def OLS_estimator(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute OLS estimators from the data.

    We use numpy broadcasting to accelerate computations
    and obtain several OLS estimators.

    Parameters:
        X: (n, d) matrix
        y: (n, n_tests) matrix

    Returns:
        theta_hat: (d, n_tests) matrix
    """
    covariance_matrix = X.T @ X
    inverse_covariance = np.linalg.inv(covariance_matrix)
    theta_hat = inverse_covariance @ (X.T @ y)
    return theta_hat


def gradient(theta, H, X, y):
    """
    Compute the gradient of the empirical risk
    as a function of theta, X, y
    for a least squares problem.

    Parameters:
        X (float matrix): (n, d) matrix
        y (float vector): (n, 1) vector
        theta (float vector): (d, 1) vector

    Returns:
        gradient of the objective function
    """
    n = y.shape[0]
    return H @ theta - 1 / n * X.T @ y
