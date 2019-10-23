import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats
import h5py
import para_est
import vis

def polynomial_evaluation(x, poly_coef=[0.15, -0.05]):
    # -0.05 * x**5 + 0.15 * x**2 + 0

    return np.polyval([poly_coef[1], 0.0, 0.0, poly_coef[0], 0.0, 0], x)

def para_est_obj(parameters, d):
    """
    An objective function for testing optimisation routines that return vectors
    """
    y = polynomial_evaluation(x, poly_coef=parameters)
    return y-d

if __name__ == '__main__':

    results_folder = './results/confidence_interval'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Generate synthetic data (d) from a fifth degree polynomial
    p = 2 # Number of parameters
    n = 101 # Number of datapoints
    poly_coef = [0.15, -0.05]
    x = np.linspace(0, 1, num=n)
    y = polynomial_evaluation(x)

    # Add noise to synthetic data
    noise_path = os.path.join(results_folder, 'noise.h5')
    generate = True
    if generate:
        mu = 0 # Mean
        sigma = 0.05 # Standard deviation
        e = np.random.normal(mu, sigma, n)
        f_handle = h5py.File(noise_path, 'w')
        f_handle.create_dataset('mu', data=mu)
        f_handle.create_dataset('sigma', data=sigma)
        f_handle.create_dataset('e', data=e)
        f_handle.close()
    else:
        f_handle = h5py.File(noise_path, 'r')
        mu = f_handle['mu'][...]
        sigma = f_handle['sigma'][...]
        e = f_handle['e'][...]
        f_handle.close()
    d = y + e

    # Estimate polynomial coefficients
    x0 = poly_coef # Initial estimate
    ps = para_est.estimation()
    ps.set_initial_parameters(np.atleast_1d(x0))
    ps.set_objective_function(para_est_obj, arguments=(d,), metric='e')
    ps.set_gtol(1e-15)
    ps.optimise()

    # Evaluate Jacobian using ndifftools
    J = ps.evaluate_derivatives_numdifftools(ps.solutions.x)
    H = ps.evaluate_derivatives(ps.solutions.x, 1e-7, evaluate='hessian')
    J = ps.evaluate_derivatives(ps.solutions.x, 1e-7, evaluate='jacobian')
    print('Initial parameters: {0}'.format(x0))
    print('Optimal parameters: {0}'.format(ps.solutions.x))
    print('Objective function: {0}'.format(ps.solutions.fun))
    print('Cost function: {0}'.format(ps.solutions.cost))
    print('Success: {0}'.format(ps.solutions.success))
    print('Message: {0}'.format(ps.solutions.message))

    # Evaluate polynomial with optimal coefficients
    opti_theta = ps.solutions.x
    y_optimal = polynomial_evaluation(x, poly_coef=opti_theta)

    # Visualise fit
    visualise = False
    if visualise:
        plt.plot(x, y, label='True polynomial')
        plt.scatter(x, d, label='Noisy data points')
        plt.plot(x, y_optimal, label='Optimal polynomial fit')
        plt.title('Synthetic data generated from a 5th degree polynomial')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([0, 1])
        plt.ylim([-0.15, 0.2])
        plt.legend(loc='lower right')
        plt.show()

    # Confidence intervals
    alpha = 0.05
    F_crit = stats.f.ppf(1.0 - alpha, p, n-p)
    chi2 = stats.chi2.ppf(1.0 - alpha, p)
    f = para_est_obj(opti_theta, d)
    S = np.dot(f, f)
    s2 = S/(n-p) # Variance
    H_approx = 2*np.dot(J.T, J)
    limit = p*s2*F_crit + S

    theta1 = np.linspace(0.0, 0.225, 100)
    theta2 = np.linspace(-0.125, 0.125, 100)
    X, Y = np.meshgrid(theta1, theta2)
    thetas = np.array([
        X.reshape(X.size),
        Y.reshape(Y.size)]).T
    obj = np.zeros(thetas.shape[0])
    obj_H_approx = np.zeros(thetas.shape[0])
    for idx, theta in enumerate(thetas):
        f = para_est_obj(theta, d)
        obj[idx] = np.dot(f, f)
        obj_H_approx[idx] = S + \
                            0.5*np.dot((theta-opti_theta).T,
                                       np.dot(H_approx, theta-opti_theta))
    Z = obj.reshape(X.shape)
    Z_H_approx = obj_H_approx.reshape(X.shape)
    interval = np.sqrt(np.diagonal(s2*np.linalg.inv(H_approx))) *\
               stats.t.ppf((1 + (1.0 - alpha))/2, n-p)

    sns.set(style="darkgrid")
    f, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect("equal")
    # Plot contour shades
    ax = vis.contour_plot(
        X, Y, Z, cmap="Blues", shade=True, shade_lowest=True, cbar=True, ax=ax)

    # Plot objective function contour lines
    ax = vis.contour_plot(
        X, Y, Z, cmap="Blues", n_levels=[limit], shade=False,
        shade_lowest=True, cbar=False, ax=ax)
    # Plot Hessian approximation lines
    ax = vis.contour_plot(
        X, Y, Z_H_approx, colors='r', n_levels=[limit], shade=False,
        shade_lowest=False, cbar=False, ax=ax)

    plt.title('Sum of squared differences objective function')
    plt.plot(
        opti_theta[0], opti_theta[1], 'ro', ms=10, label='identified solution')
    plt.plot(poly_coef[0], poly_coef[1], 'bo', ms=10, label='true solution')
    plt.errorbar(
        opti_theta[0], opti_theta[1], xerr=interval[0], yerr=interval[1],
        capthick=5)
    plt.legend(loc='lower left', numpoints=1)
    plt.xlabel('theta1')
    plt.ylabel('theta2')
    plt.show()

    a = 1

    initial_parameters = [1.0]
    # Define target data e.g. node x,y,z positions in prone or supine
    num_evaluation_positions = 5000
    target_positions = np.zeros([num_evaluation_positions, 3])
