import pickle
import unittest

import numpy as np

from ilurl.networks.base import Network

# incoming approaches
INCOMING_247123161 = {0: [('309265401', 0), ('309265401', 1), ('-238059328', 0), ('-238059328', 1)],
                      1: [('383432312', 0), ('-238059324', 0)]}

INCOMING_247123464 = {0: [('309265400', 0), ('309265400', 1), ('-309265401', 0), ('-309265401', 1)],
    1: [('22941893', 0)]}
INCOMING_247123468 = {0: [('309265402', 0), ('309265402', 1), ('-309265400', 0), ('-309265400', 1)], 1: [('23148196', 0)]}

# internal outgoing approaches; lanes between two tl
OUTGOING_247123161 = [('-309265401', 0), ('-309265401', 1)]
OUTGOING_247123464 = [('-309265400', 0), ('-309265400', 1), ('309265401', 0), ('309265401', 1)]
OUTGOING_247123468 = [('309265400', 0), ('309265400', 1)]

MAX_VEHS = {('247123161', 0): 36,
            ('247123161', 1): 16,
            ('247123464', 0): 32,
            ('247123464', 1): 9,
            ('247123468', 0): 32,
            ('247123468', 1): 9}


MAX_VEHS_OUT = {('247123161', 0): 16,
                ('247123161', 1): 16,
                ('247123464', 0): 34,
                ('247123464', 1): 34,
                ('247123468', 0): 16,
                ('247123468', 1): 16}

class TestGridBase(unittest.TestCase):
    """
        * Defines a network and the kernel data.

        * Common base for problem formulation tests.
    """
    def setUp(self):

        network_args = {
            'network_id': 'grid',
            'horizon': 999,
            'demand_type': 'constant',
            'tls_type': 'rl'
        }
        self.network = Network(**network_args)

        with open('tests/data/grid_kernel_data.dat', "rb") as f:
            kernel_data = pickle.load(f)
        self.kernel_data = kernel_data

        with open('tests/data/grid_kernel_data_1.dat', "rb") as f:
            kernel_data_1 = pickle.load(f)
        self.kernel_data_1 = kernel_data_1


        with open('tests/data/grid_kernel_data_2.dat', "rb") as f:
            kernel_data_2 = pickle.load(f)
        self.kernel_data_2 = kernel_data_2

    def test_kernel_data(self):
        self.assertEqual(len(self.kernel_data), 60)

    def test_kernel_data_1(self):
        self.assertEqual(len(self.kernel_data_1), 60)

    def test_kernel_data_2(self):
        self.assertEqual(len(self.kernel_data_2), 60)

if __name__ == '__main__':
    unittest.main()
