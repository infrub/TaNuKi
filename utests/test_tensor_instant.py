import unittest
import sys
sys.path.append('../')
from tanuki import *
import numpy as np


class TestTensorInstant(unittest.TestCase):
    def test_instant(self):
        a = random_tensor((2,3,4))
        self.assertEqual(a.shape, (2,3,4))
        a = zeros_tensor((2,3,4))
        self.assertEqual(a.shape, (2,3,4))
        a = identity_tensor((2,3))
        self.assertEqual(a.shape, (2,3,2,3))
        a = identity_tensor((2,3),6)
        self.assertEqual(a.shape, (2,3,6))





if __name__=="__main__":
    unittest.main()