import numpy as np
from scipy.optimize import least_squares

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

    def set_objective_function(self, objective_function, arguments=None):
        """
        Set the objective function. Add additional input variables as a tuple
        in the arguments option .e.g arguments=(variable1, variable2). If there
        is only 1 variable then set as arguments=(variable1,). The comma is
        important to define a tuple array.
        """
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

    def evaluate_hessian(self, x, stepsize, optimal_design=False):
        """
        Routine for evaluating the Hessian matrix using central finite differences
        """

        def objfun(x):
            # Return sum of squares objective function
            return np.sum(self.objective_function(x, *self.objective_function_arguments) ** 2)

        n = len(x)
        A = np.zeros(n)
        B = np.zeros(n)
        ee = stepsize * np.eye(n)

        # First-order derivatives: 2n function calls needed
        J = np.zeros((n))
        for i in range(n):
            A[i] = objfun(x + ee[:, i])
            B[i] = objfun(x - ee[:, i])
            J[i] = (A[i]-B[i])/2.0*ee[i, i]

        # Second-order derivatives based on function calls only (Abramowitz and Stegun 1972, p.884): for dense Hessian, 2n+4n^2/2 function calls needed.

        if optimal_design:
            E = 0.0
        else:
            E = objfun(x)

        H = np.zeros((n, n))
        for i in range(n):
            C = objfun(x + 2 * ee[:, i])
            F = objfun(x - 2 * ee[:, i])
            H[i, i] = (- C + 16 * A[i] - 30 * E + 16 * B[i] - F) / (12 * (ee[i, i] ** 2))
            for j in range(i + 1, n):
                G = objfun(x + ee[:, i] + ee[:, j])
                I = objfun(x + ee[:, i] - ee[:, j])
                L = objfun(x - ee[:, i] + ee[:, j])
                K = objfun(x - ee[:, i] - ee[:, j])
                H[i, j] = (G - I - L + K) / (4 * ee[i, i] * ee[j, j])
                H[j, i] = H[i, j]

        import cmath
        n = len(H)
        detH = np.linalg.det(H)
        condH = 1.0 / np.linalg.cond(H)
        H0 = np.zeros((n, n))
        for j in range(n):
            for k in range(n):
                H0[j, k] = H[j, k] / (cmath.sqrt(H[j, j] * H[k, k]))
        detH0 = np.linalg.det(H0)
        return H, detH, condH, detH0, J




