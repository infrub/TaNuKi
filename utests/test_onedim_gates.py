import unittest
import sys
sys.path.append('../')
from tanuki import *
from tanuki.onedim import *
import numpy as np
from math import sqrt
import copy

class TestGates(unittest.TestCase):
    def test_01(self):
        G = zeros_tensor((2,2,2,2), ["out0","out1","in0","in1"])
        G.data[0,0,0,0] = 1.5
        G.data[0,1,0,1] = -1.5
        G.data[1,0,1,0] = -1.5
        G.data[1,1,1,1] = 1.5
        G = OneDimGate(G, [["out0"],["out1"]],[["in0"],["in1"]])
        G = G.exp(-2j)
        F = zeros_tensor((2,2,2,2), ["out0","out1","in0","in1"])
        F.data[0,0,0,0] = np.exp(-3j)
        F.data[0,1,0,1] = np.exp(3j)
        F.data[1,0,1,0] = np.exp(3j)
        F.data[1,1,1,1] = np.exp(-3j)
        self.assertEqual(G.tensor, F)
        self.assertTrue(G.tensor.is_unitary(["out0","out1"]))
        G = G.to_fin1DSimBTPO()
        self.assertEqual(len(G),2)




if __name__=="__main__":
    unittest.main()