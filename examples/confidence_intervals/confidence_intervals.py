"""Example for calculating confidence intervals after parameter estimation.

Authors: Thiranja Prasad Babarenda Gamage
Auckland Bioengineering Institute.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
from scipy import stats
import h5py
import para_est
import vis

def polynomial_evaluation(x, poly_coef=None):
    # -0.05 * x**5 + 0.15 * x**2 + 0
    all_poly_coef = [0.15, -0.05]
    if poly_coef is not None:
        for idx, p in enumerate(poly_coef):
            all_poly_coef[idx] = p

    return np.polyval(
        [all_poly_coef[1], 0.0, 0.0, all_poly_coef[0], 0.0, 0], x)

def para_est_obj(parameters, x, d):
    """Parameter estimation.
    """
    y = polynomial_evaluation(x, poly_coef=parameters)
    return y-d

def main(cfg, test=False):

    # Headless plotting.
    matplotlib.use('Agg')

    results_folder = './results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Generate synthetic data (d) from a fifth degree polynomial
    p = 1 # Number of parameters
    n = 201 # Number of datapoints
    poly_coef = [0.15, -0.05]
    x = np.linspace(0, 1, num=n)
    y = polynomial_evaluation(x)

    # Add noise to synthetic data
    noise_path = os.path.join(results_folder, 'noise.h5')
    if test:
        f_handle = h5py.File(noise_path, 'r')
        mu = f_handle['mu'][...]
        sigma = f_handle['sigma'][...]
        e = f_handle['e'][...]
        f_handle.close()
    else:
        mu = 0 # Mean
        sigma = 0.05 # Standard deviation
        e = np.random.normal(mu, sigma, n)
        f_handle = h5py.File(noise_path, 'w')
        f_handle.create_dataset('mu', data=mu)
        f_handle.create_dataset('sigma', data=sigma)
        f_handle.create_dataset('e', data=e)
        f_handle.close()
    d = y + e

    # Estimate polynomial coefficients
    x0 = poly_coef[:p] # Initial estimate
    ps = para_est.estimation()
    ps.set_initial_parameters(np.atleast_1d(x0))
    ps.set_objective_function(para_est_obj, arguments=(x, d,), metric='e')
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
    visualise = True
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
    alpha95 = 0.05
    alpha90 = 0.1

    f = para_est_obj(opti_theta, x, d)
    S = np.dot(f, f)
    s2 = S/(n-p) # Variance
    H_approx = 2*np.dot(J.T, J)

    def calc_limit(p, n, alpha, S):
        F_crit = stats.f.ppf(1.0 - alpha, p, n - p)
        # chi2 = stats.chi2.ppf(1.0 - alpha, p)
        return p * s2 * F_crit + S

    limit95 = calc_limit(p, n, alpha95, S)
    limit90 = calc_limit(p, n, alpha90, S)

    # Define parameters for sweeps.
    theta1 = np.linspace(0.0, 0.225, 100)
    if p == 1:
        # Represent as a 2D array e.g. [[1],[2],...,[3]].
        thetas = theta1.reshape((-1, 1))
    elif p == 2:
        theta2 = np.linspace(-0.125, 0.125, 100)
        X, Y = np.meshgrid(theta1, theta2)
        thetas = np.array([
            X.reshape(X.size),
            Y.reshape(Y.size)]).T

    # Evaluate objective function at each parameter sweep point - direct
    # evaluation and approximation from the Hessian.
    obj = np.zeros(thetas.shape[0])
    obj_H_approx = np.zeros(thetas.shape[0])
    for idx, theta in enumerate(thetas):
        f = para_est_obj(theta, x, d)
        obj[idx] = np.dot(f, f)
        obj_H_approx[idx] = S + \
                            0.5*np.dot((theta-opti_theta).T,
                                       np.dot(H_approx, theta-opti_theta))
    if p == 1:
        Z = obj
        Z_H_approx = obj_H_approx
    elif p == 2:
        Z = obj.reshape(X.shape)
        Z_H_approx = obj_H_approx.reshape(X.shape)

    # Visualise objective function.
    sns.set(style="darkgrid")
    f, ax = plt.subplots(figsize=(10, 8))
    plt.xlabel('theta1')
    plt.xlim([min(theta1), max(theta1)])
    if p == 1:
        plt.ylabel('SSD')
        plt.ylim([0, max(Z)])

        import pandas as pd
        df = pd.DataFrame()
        df['theta1'] = theta1
        df['SSD'] = Z
        df['SSD_H_approx'] = Z_H_approx
        df['95% CI'] = np.ones_like(theta1)*limit95
        df['90% CI'] = np.ones_like(theta1)*limit90

        from scipy.spatial import cKDTree
        tree = cKDTree(Z.reshape((-1, 1)))
        neighbours = 2
        idxs95 = tree.query([limit95], k=neighbours)[1]
        idxs90 = tree.query([limit90], k=neighbours)[1]

        plt.plot(
            theta1, Z, 'r', label='Objective function')
        plt.plot(theta1, Z_H_approx, 'g--',
                 label='Objective function from H approx')
        plt.plot(theta1, df['95% CI'], 'k-',
                 label='95% CI')
        plt.plot(theta1, df['90% CI'], 'k--',
                 label='90% CI')
        plt.plot(np.ones_like(theta1)*theta1[idxs95[0]], np.linspace(0, max(Z), len(theta1)), 'k-')
        plt.plot(np.ones_like(theta1)*theta1[idxs95[1]], np.linspace(0, max(Z), len(theta1)), 'k-')
        plt.plot(np.ones_like(theta1)*theta1[idxs90[0]], np.linspace(0, max(Z), len(theta1)), 'k--')
        plt.plot(np.ones_like(theta1)*theta1[idxs90[1]], np.linspace(0, max(Z), len(theta1)), 'k--')
        plt.plot(
            opti_theta[0], S, 'ro', ms=10, label='identified solution')
        plt.plot(poly_coef[0], 0, 'bo', ms=10,
                 label='true solution')
        #ax.axhline(limit)
    elif p == 2:
        ax.set_aspect("equal")
        plt.ylabel('theta2')
        plt.ylim([min(theta2), max(theta2)])

        # Plot contour shades
        ax = vis.contour_plot(
            X, Y, Z, cmap="Blues", shade=True, shade_lowest=True, cbar=True, ax=ax)

        # Plot objective function contour lines
        ax = vis.contour_plot(
            X, Y, Z, colors='g', n_levels=[limit95], shade=False,
            shade_lowest=True, cbar=False, ax=ax, linewidths=4)
        # Plot Hessian approximation lines
        ax = vis.contour_plot(
            X, Y, Z_H_approx, colors='r', n_levels=[limit95], shade=False,
            shade_lowest=False, cbar=False, ax=ax, linestyles='dashed',
            linewidths=4)
        plt.plot(
            opti_theta[0], opti_theta[1], 'ro', ms=10,
            label='identified solution')
        plt.plot(poly_coef[0], poly_coef[1], 'bo', ms=10,
                 label='true solution')

    plt.title('Sum of squared differences objective function')
    plt.legend(loc='lower left', numpoints=1)
    plt.savefig(os.path.join(results_folder, 'parameter_sweep.png'))
    plt.show()
    plt.show()
    plt.clf()

if __name__ == '__main__':
    main(None, test=False)
