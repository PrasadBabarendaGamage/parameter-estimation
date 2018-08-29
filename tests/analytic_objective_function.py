import unittest
import para_est
import numpy as np

def fun_rosenbrock_vector(x):
    """
    An objective function for testing optimisation routines that return vectors
    """
    return np.array([10*(x[1]-x[0])**2,1-x[0]])

def fun_rosenbrock_scalar(x):
    """
    An objective function for testing optimisation routines that return scalars
    """
    return 100.0*(x[1]-x[0]**2.0)**2.0 + (1-x[0])**2.0

def fun_rosenbrock_hessian(x):
    # type: (object) -> object
    """
    Analytic hessian of the rosenbrock function
    """
    return np.array([[1200*x[0]**2 - 400*x[1] + 2, -400*x[0]],
                 [-400*x[0], 200]])

class Parameter_estimation_tests(unittest.TestCase):
    def test_estimation(self):
        """
        Test the parameter estimation routines
        """
        ps = para_est.estimation()
        ps.set_initial_parameters(np.array([-1.9, 2]))
        ps.set_objective_function(fun_rosenbrock_vector)
        ps.optimise()
        ps.set_objective_function(fun_rosenbrock_scalar)
        H, detH, condH, detH0 = ps.evaluate_hessian(ps.solutions.x, 1.e-7)
        print(ps.solutions.x)
        print('Finite difference Hessian')
        print(H)
        analytic_H = fun_rosenbrock_hessian(ps.solutions.x)
        print('analytic Hessian')
        print(analytic_H)
        J = ps.solutions.jac
        quasi_hessian_approx = 2 * (J.T.dot(J))
        print('quasi-Newton Hessian_approx')
        print(quasi_hessian_approx)
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
