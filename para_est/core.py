import numpy as np
from scipy.optimize import least_squares

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
            self, objective_function, arguments=None, metric='sse',
            return_type='scalar'):
        """
        Set the objective function. Add additional input variables as a tuple
        in the arguments option .e.g arguments=(variable1, variable2). If there
        is only 1 variable then set as arguments=(variable1,). The comma is
        important to define a tuple array.
        """
        self.metric = metric
        self.return_type = return_type
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
                                       diff_step=1e-5)
        return self.solutions

    def evaluate_derivatives(
            self, x, stepsize, optimal_design_no_noise=False, debug=False, 
            quasi_newton=False, method='in-built',
            functions=('gradient', 'jacobian', 'hessian')):
        """
        Routine for evaluating the Hessian matrix using central finite differences
        Jacobian matrix of a scalar function is just the gradient


        """

        def objfun(x):
            obj = self.objective_function(x,
                                          *self.objective_function_arguments)
            obj = postprocess_objfun(obj, self.metric, self.return_type, method)
            return obj

        if method == 'in-built':
            g = None
            h = _compute_absolute_step(stepsize, x)
            h_vecs = np.diag(h)
    
            n = len(x)
            A = np.zeros(n)
            B = np.zeros(n)
    
            # First-order derivatives: 2n function calls needed
            J = np.zeros((n))
            for i in range(n):
                
                if debug:
                    x1 = x - h_vecs[i]
                    x2 = x + h_vecs[i]
                    dx = x2[i] - x1[i]
                    f1 = objfun(x1)
                    f2 = objfun(x2)
                    df = f2 - f1
                
                A[i] = objfun(x + h_vecs[:, i])
                B[i] = objfun(x - h_vecs[:, i])
                J[i] = (A[i]-B[i])/(2.0*h_vecs[i, i])
    
            if quasi_newton:
                pass
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

        elif method == 'ndifftools':
            import numdifftools as nd
            J = nd.Jacobian(objfun, method='central')(x)
            g = nd.Gradient(objfun, method='central')(x)
            #H = nd.Hessian(objfun, method='central')(x)
            H=None
        else:
            raise ValueError(
                'Selected method for computing Hessian is not supported')
        return H, J, g

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

def postprocess_objfun(obj, metric, return_type, method):
    if metric == 'sse':
        if return_type == 'scalar':
            pass
        else:
            raise ValueError('SSE objective function that returns a '
                             'vector is not supported')
    elif metric == 'e':
        if method == 'in-built':
            if return_type == 'vector' or \
                    return_type == 'scalar':
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

