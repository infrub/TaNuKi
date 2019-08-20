import unittest
import sys
sys.path.append('../')
from tanuki import *
from tanuki.onedim import *
import numpy as np
from math import sqrt
import copy

class TestProduct(unittest.TestCase):
    def test_inner(self):
        A = random_fin1DSimTPS([["p0"],["p1"],["p2"]])
        B = random_fin1DSimTPS([["q0"],["q1"],["q2"]])
        a = A.to_tensor()
        b = B.to_tensor()
        C = inner_product_fin1DSimTPS_fin1DSimTPS(A,B)
        c = a.conj()[["p0","p1","p2"]] * b["q0","q1","q2"]
        self.assertEqual(C,c)

    def test_abs_sub(self):
        phys_labelss = [["p0"],["p1"],["p2"]]
        A = random_fin1DSimTPS(phys_labelss)
        B = random_fin1DSimTPS(phys_labelss)
        C = abs_sub_fin1DSimTPS_fin1DSimTPS(A,B)
        a = A.to_tensor()
        b = B.to_tensor()
        c = (a - b).norm()
        self.assertAlmostEqual(C,c)






if __name__=="__main__":
    unittest.main()