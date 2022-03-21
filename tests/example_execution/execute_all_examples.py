"""Tests execution of all module examples.

Tests should run as fast as possible to enable fast feedback during code
development. This test script aims to only test the execution of examples
e.g. to check for runtime errors if the module's api was changed,
but the exmaple has not yet been updated accordingly.

The directory of each example and the name of the example script to test needs
to be added to the example_list global variable.

No two examples can have the same script name. Note that if a test fails, then
the data structures, e.g. OpenCMISS objects, may not be finalised. Subsequent
tests may fail if they use the same data structures. it is therefore important
to address any test issues in numerical order. To address this issue, proper
cleanup of data structures, e.g. through a callback, is required whenever
an arbitrary error is encountered. This has not yet been implemented in the
module.

Authors: Thiranja Prasad Babarenda Gamage
Organisation: Auckland Bioengineering Institute, University of Auckland
"""

import os
import sys
import json
import unittest
from parameterized import parameterized


# Ignore tensorflow FutureWarning messages.
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow

# Load in the metadata for the stf dataset that will be used for testing.
config_file_path = '../../study_configurations/stf.config'
with open(config_file_path) as config_file:
    cfg = json.load(config_file)
config_file.close()

example_root_directory = os.path.abspath('../../examples')

# List of examples to test [example directory, example name].
# Note that some of these examples need to be run in order (e.g. output from
# 'machine_learning_mechanics/configuration_generator/' is required for most
# of the subsequent machine learning mechanics examples.
example_list = [
    ['confidence_intervals/', 'confidence_intervals']
]

# Postprocessing of example_list to address peculiarities in parameterized
# python module.

# "parameterized" module uses the first array value in the example_list as the
# test name. Reverse order of example directory and example name to allow the
# example name to be used as the test name.
example_list = [example[::-1] for example in example_list[::-1]]

# "parameterized" module runs through tests in reverse order. Reverse order of
# example list such that the examples run in the order as listed in the
# example list above.
example_list = example_list[::-1]

class TestExampleExecution(unittest.TestCase):
    """Class for testing execution of all examples in the example list.

    Note that each example needs to have a main() function. To achieve
    efficient testing, a test=True input argument is passed to the main()
    function. The user can use this to bypass time consuming tasks e.g. for
    mechanics, it can be used to skip the mechanics solves and quickly test the
    infrastructure. Other tests are necessary to verify/validate that the
    mechanics output is correct e.g. comparisons to expected output/analytic
    solutions.
    """
    @parameterized.expand(example_list)
    def test(self, example_name, example_dir):
        """Test execution of the specified heart_mech example script.

        The examples are run in their original directories. They have been
        configured to export any output to 'results_test' folder in their
        original directories.

        Args:
            example_name: Name of example to be tested.
            example_dir: Directory name for the example.
        """

        os.chdir(os.path.join(example_root_directory, example_dir))

        # Add example directory (now the current working directory) to python
        # sys path.
        sys.path.insert(0, './')

        # Import example script.
        example = __import__(example_name)

        # Execute example.
        example.main(cfg, test=True)

        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
