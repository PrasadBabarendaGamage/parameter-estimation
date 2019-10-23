""" Test Hessian matrix evaluation

Tests Hessian matrix evaluation from finite differences using the Rosenbrock's
"banana function".
https://www.mathworks.com/help/optim/ug/banana-function-minimization.html
"""

import unittest
import para_est
import numpy as np
import numdifftools as nd
import numdifftools.nd_scipy as ndscipy

class Parameter_estimation_tests(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        """
        Setup the the cantilever simulation object and store results in
        class attributes for use in subsequent tests.
        """
        self.test_x = np.array([2., 1.])
        self.fd_step = 1.4901161193847656e-5
        self.ps = para_est.estimation()
        self.ps.set_initial_parameters(self.test_x)

    @classmethod
    def tearDownClass(self):
        pass

    def test_gradient(self):
        print('Test Gradient')
        x = self.test_x
        analytic_solution = para_est.Rosenbrock()
        h = self.fd_step

        # Analytic
        g_analytic = analytic_solution.g(x)

        # numdifftools
        g_ndifftools = nd.Gradient(analytic_solution.f_scalar)(x)
        g_scipy = ndscipy.Gradient(analytic_solution.f_scalar, h)(x)

        # para_est
        self.ps.set_objective_function(analytic_solution.f_vector, metric='e')
        g_ps = self.ps.evaluate_derivatives(x, h, evaluate='gradient')

        # Log
        print(' g analytic  : {0}'.format(g_analytic))
        print(' g_ndifftools: {0}'.format(g_ndifftools))
        print(' g scipy     : {0}'.format(g_scipy))
        print(' g ps        : {0}'.format(g_ps))
        print(' ps function evaluation count: {0}'.format(
            self.ps.logger.get_f_eval_count()))

        self.assertEqual(
            True, np.all(np.isclose(
                g_ps, g_analytic, rtol=1.e-14, atol=1.e-2)))

    def test_jacobian(self):
        print('Test Jacobian')
        x = self.test_x
        analytic_solution = para_est.Rosenbrock()
        h = self.fd_step

        # Analytic
        j_analytic = analytic_solution.j(x)

        # numdifftools
        j_ndifftools = nd.Jacobian(analytic_solution.f_vector)(x)
        j_scipy = ndscipy.Jacobian(analytic_solution.f_vector, h)(x)

        # para_est
        self.ps.set_objective_function(analytic_solution.f_vector, metric='e')
        j_ps = self.ps.evaluate_derivatives(x, h, evaluate='jacobian')

        # Log
        print(' j analytic  : {0}'.format(j_analytic))
        print(' j_ndifftools: {0}'.format(j_ndifftools))
        print(' j scipy     : {0}'.format(j_scipy))
        print(' j ps        : {0}'.format(j_ps))
        print(' ps function evaluation count: {0}'.format(
            self.ps.logger.get_f_eval_count()))

        self.assertEqual(
            True, np.all(np.isclose(
                j_ps, j_analytic, rtol=1.e-14, atol=1.e-2)))

    def test_hessian_full(self):
        print('Test full Hessian')
        x = self.test_x
        analytic_solution = para_est.Rosenbrock()
        h = self.fd_step

        # Analytic
        h_analytic = analytic_solution.h(x)

        # ndifftools scalar
        h_ndifftools = nd.Hessian(analytic_solution.f_scalar)(x)

        # para_est vector
        self.ps.set_objective_function(analytic_solution.f_vector, metric='e')
        h_ps_vec = self.ps.evaluate_derivatives(x, h, evaluate='hessian')

        # para_est scalar
        self.ps.set_objective_function(analytic_solution.f_scalar, metric='sse')
        h_ps_sca = self.ps.evaluate_derivatives(x, h, evaluate='hessian')

        # Log
        print(' h analytic  : {0}'.format(h_analytic.flatten()))
        print(' h_ndifftools: {0}'.format(h_ndifftools.flatten()))
        print(' h ps vector : {0}'.format(h_ps_vec.flatten()))
        print(' h ps scalar : {0}'.format(h_ps_sca.flatten()))
        print(' ps function evaluation count: {0}'.format(
            self.ps.logger.get_f_eval_count()))

        self.assertEqual(
            True, np.all(np.isclose(h_ps_sca.flatten(), h_analytic.flatten(),
                                    rtol=1.e-14, atol=1.e-2)))

    def test_hessian_quasi_newton(self):
        print('Test quasi-Newton Hessian')
        x = self.test_x
        analytic_solution = para_est.Rosenbrock()
        h = self.fd_step

        # Analytic
        h_analytic = analytic_solution.h(x)

        # para_est vector
        self.ps.set_objective_function(analytic_solution.f_vector, metric='e')
        h_ps = self.ps.evaluate_derivatives(
            x, h, evaluate='hessian', quasi_newton=True)

        # Log
        print(' h analytic  : {0}'.format(h_analytic.flatten()))
        print(' h ps        : {0}'.format(h_ps.flatten()))
        print(' ps function evaluation count: {0}'.format(
            self.ps.logger.get_f_eval_count()))

        self.assertEqual(
            True, np.all(np.isclose(h_ps.flatten(), h_analytic.flatten(),
                                    rtol=1.e-14, atol=1.e-2)))

if __name__ == '__main__':
    unittest.main()
