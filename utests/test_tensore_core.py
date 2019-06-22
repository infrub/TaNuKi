import unittest
import sys
sys.path.append('../')
from tanuki import *
import numpy as np


class TestTensor(unittest.TestCase):
    def test_basic(self):
        a = Tensor([[1,2],[3,4]],["a","b"])


if __name__=="__main__":
    unittest.main()