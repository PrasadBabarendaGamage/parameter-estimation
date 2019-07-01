import numpy as np
import para_est

def objective_function(parameters, target_positions):
    """
    An objective function for testing optimisation routines that return vectors
    """
    # Solve model using input parameters to determine model predicted positions
    num_evaluation_positions = 5000
    model_positions = np.zeros([num_evaluation_positions, 3])

    # Calculate error between prediction and target (note this should be a Nx3
    # array, i.e. the x,y,z error at each of the N=num_evaluation_positions.

    error = model_positions - target_positions

    # Return error as a flattened error
    return error.flatten()


if __name__ == '__main__':

    initial_parameters = [1.0]
    # Define target data e.g. node x,y,z positions in prone or supine
    num_evaluation_positions = 5000
    target_positions = np.zeros([num_evaluation_positions, 3])

    # Setup a estimation object
    ps = para_est.estimation()
    ps.set_initial_parameters(np.atleast_1d(initial_parameters))
    ps.set_objective_function(objective_function, arguments=(target_positions,))
    ps.optimise()

    # Display results
    print('Initial parameters: {0}'.format(initial_parameters))
    print('Optimal parameters: {0}'.format(ps.solutions.x))
    print('Objective function at optimal parameters: {0}'.format(ps.solutions.fun))
    print('Cost function: {0}'.format(ps.solutions.cost))
    print('Success: {0}'.format(ps.solutions.success))
    print('Message: {0}'.format(ps.solutions.message))

