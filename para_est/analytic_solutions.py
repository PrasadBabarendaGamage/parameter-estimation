import numpy as np

class Rosenbrock():
    """
    Rosenbrock function and its derivatives
    """
    def f_scalar(self, x):
        return 100.0 * (x[1] - x[0] ** 2.0) ** 2 + (1 - x[0]) ** 2

    def f_vector(self, x):
        return np.array([10 * (x[1] - x[0] ** 2),
                         1 - x[0]])

    def g(self, x):
        return np.array([-400 * (x[1] - x[0] ** 2) * x[0] - 2 * (1 - x[0]),
                         200 * (x[1] - x[0] ** 2)])

    def j(self, x):
        return np.array([[-20 * x[0], 10],
                         [-1, 0]])

    def h(self, x):
        return np.array([[1200 * x[0] ** 2 - 400 * x[1] + 2, -400 * x[0]],
                         [-400 * x[0], 200]])
