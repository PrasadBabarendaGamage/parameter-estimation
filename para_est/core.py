import numpy as np
from scipy.optimize import least_squares
import para_est

EPS = np.finfo(np.float64).eps
relative_step = {"2-point": EPS ** 0.5,
                 "3-point": EPS ** (1 / 3),
                 "cs": EPS ** 0.5}

class estimation:
    """
    Parameter estimation class
    """

    def __init__(self):
        """
        Create a new parameter estimation instance with no defaults.
        """
        self.lower_bounds = -np.inf
        self.upper_bounds = np.inf
        self.initial_parameters = None
        self.objective_function = None
        self.objective_function_arguments = ()
        self.solutions = None
        self.logger = para_est.Logger()
        self.ftol = 1e-8
        self.xtol = 1e-8
        self.gtol = 1e-8

    def set_upper_bounds(self,upper_bounds=np.inf):
        """
        Set upper bounds. Defaults to upper_bounds=np.inf ie not bounds. 
        Bounds need to be specified as a numpy 1D array with a length equal 
        to the number of parameters being identified. 
        """
        self.upper_bounds = upper_bounds


    def set_lower_bounds(self,lower_bounds=-np.inf):
        """
        Set lower bounds. Defaults to lower_bounds=np.inf ie not bounds. 
        Bounds need to be specified as a numpy 1D array with a length equal 
        to the number of parameters being identified. 
        """
        self.lower_bounds = lower_bounds

    def set_initial_parameters(self,initial_parameters):
        """
        Set initial parameters (numpy array)
        """
        self.initial_parameters = initial_parameters

    def set_objective_function(
            self, objective_function, arguments=None, metric='sse'):
        """
        Set the objective function. Add additional input variables as a tuple
        in the arguments option .e.g arguments=(variable1, variable2). If there
        is only 1 variable then set as arguments=(variable1,). The comma is
        important to define a tuple array.
        Return type (e.g. scalar or vector will be infered)
        """
        self.metric = metric
        if callable(objective_function):
            self.objective_function = objective_function
        else:
            raise TypeError('The objective function must be callable ie it '
                            'must be a function')
        if arguments is not None:
            if isinstance(arguments, tuple):
                self.objective_function_arguments = arguments
            else:
                raise TypeError('The objective function arguments must be a '
                                'tuple')

    def set_ftol(self, ftol):
        """
        Set function tolerance
        """
        self.ftol = ftol

    def set_gtol(self, gtol):
        """
        Set gradient tolerance
        """
        self.gtol = gtol

    def set_xtol(self, xtol):
        """
        Set parameter tolerance
        """
        self.xtol = xtol

    def optimise(self):
        """
        Routine for running the optimisation
        """
        if self.initial_parameters is None:
            raise ValueError('Initial parameters need to be defined')
        if self.objective_function is None:
            raise ValueError('An objective function need to be set')

        self.solutions = least_squares(self.objective_function,
                                       self.initial_parameters,
                                       args=self.objective_function_arguments,
                                       bounds=(self.lower_bounds,
                                               self.upper_bounds),
                                       diff_step=1e-5, xtol=self.xtol,
                                       gtol=self.gtol, ftol=self.ftol)
        return self.solutions

    def evaluate_derivatives(
            self, x, stepsize, optimal_design_no_noise=False, debug=False, 
            quasi_newton=False, evaluate='hessian'):
        """
        Routine for evaluating the Hessian matrix using central finite differences
        Jacobian matrix of a scalar function is just the gradient


        """
        def objfun(x, return_type='scalar'):
            obj = self.objective_function(x,
                                          *self.objective_function_arguments)
            obj = postprocess_objfun(obj, self.metric, return_type,
                                     method='in-built')
            self.logger.increment()
            return obj

        self.logger.reset()

        h = _compute_absolute_step(stepsize, x)
        h_vecs = np.diag(h)

        n = len(x)
        A = np.zeros(n)
        B = np.zeros(n)

        # First-order derivatives: 2n function calls needed
        J = []
        g = np.zeros(n)
        for i in range(n):

            if debug:
                x1 = x - h_vecs[i]
                x2 = x + h_vecs[i]
                dx = x2[i] - x1[i]
                f1 = objfun(x1)
                f2 = objfun(x2)
                df = f2 - f1

            A_vec = objfun(x + h_vecs[:, i], return_type='vector')
            B_vec = objfun(x - h_vecs[:, i], return_type='vector')
            if self.metric == 'e':
                A[i] = np.sum(A_vec ** 2)
                B[i] = np.sum(B_vec ** 2)
            elif self.metric == 'sse':
                A[i] = A_vec
                B[i] = B_vec
            g[i] = (A[i]-B[i])/(2.0*h_vecs[i, i])
            J.append((A_vec-B_vec)/(2.0*h_vecs[i, i]))
        J = np.array(J).T
        if evaluate == 'gradient':
            return g
        elif evaluate == 'jacobian':
            return J
        elif evaluate == 'hessian':
            if quasi_newton:
                return 2.0*np.dot(J.T, J)
            else:
                # Second-order derivatives based on function calls only
                # (Abramowitz and Stegun 1972, p.884): for dense Hessian,
                # 2n+4n^2/2 function calls needed.

                if optimal_design_no_noise:
                    E = 0.0
                else:
                    E = objfun(x)

                H = np.zeros((n, n))
                for i in range(n):
                    C = objfun(x + 2 * h_vecs[:, i])
                    F = objfun(x - 2 * h_vecs[:, i])
                    H[i, i] = (- C + 16 * A[i] - 30 * E + 16 * B[i] - F) / (12 * (h_vecs[i, i] ** 2))
                    for j in range(i + 1, n):
                        G = objfun(x + h_vecs[:, i] + h_vecs[:, j])
                        I = objfun(x + h_vecs[:, i] - h_vecs[:, j])
                        L = objfun(x - h_vecs[:, i] + h_vecs[:, j])
                        K = objfun(x - h_vecs[:, i] - h_vecs[:, j])
                        H[i, j] = (G - I - L + K) / (4 * h_vecs[i, i] * h_vecs[j, j])
                        H[j, i] = H[i, j]
            return H



    def evaluate_derivatives_numdifftools(
            self, x, functions=('gradient', 'jacobian', 'hessian'),
            method='central'):
        import numdifftools as nd

        self.logger.reset()

        def objfun_scalar(x):
            obj = self.objective_function(
                x, *self.objective_function_arguments)
            obj = postprocess_objfun(
                obj, self.metric, return_type='scalar', method='ndifftools')
            self.logger.increment()
            return obj

        def objfun_vector(x):
            obj = self.objective_function(
                x, *self.objective_function_arguments)
            obj = postprocess_objfun(
                obj, self.metric, return_type='vector', method='ndifftools')
            self.logger.increment()
            return obj

        # Todo re-order outputs based on functions input
        g = []
        J = []
        H = []
        if 'gradient' in functions:
            g = nd.Gradient(objfun_scalar, method=method)(x)
        if 'jacobian' in functions:
            J = nd.Jacobian(objfun_vector, method=method)(x)
        if 'hessian' in functions:
            H = nd.Hessian(objfun_scalar, method=method)(x)
        return g, J, H

def identifiability_metrics(H):

    import cmath
    n = len(H)
    detH = np.linalg.det(H)
    condH = 1.0 / np.linalg.cond(H)
    H0 = np.zeros((n, n))
    for j in range(n):
        for k in range(n):
            H0[j, k] = H[j, k] / (cmath.sqrt(H[j, j] * H[k, k]))
    detH0 = np.linalg.det(H0)
    return detH, condH, detH0

def postprocess_objfun(obj, metric, return_type='scalar', method='in-built'):
    if metric == 'sse':
        if np.isscalar(obj):
            pass
        else:
            raise ValueError('SSE objective function that returns a '
                             'vector is not supported')
    elif metric == 'e':
        if method == 'in-built':
            if return_type == 'vector':
                pass
            elif return_type == 'scalar':
                obj = np.sum(obj ** 2)
        elif method == 'ndifftools':
            if return_type == 'vector':
                obj = np.array([obj]).T
            elif return_type == 'scalar':
                obj = np.sum(obj ** 2)
    return obj

def _compute_absolute_step(rel_step, x0, method="cs"):
    x0 = np.array(x0)
    if rel_step is None:
        rel_step = relative_step[method]
    sign_x0 = (x0 >= 0).astype(float) * 2 - 1
    return rel_step * sign_x0 * np.maximum(1.0, np.abs(x0))

