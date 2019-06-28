import unittest
import sys
sys.path.append('../')
from tanuki import *
from tanuki.netcon import *
import numpy as np
from math import sqrt
import time



class TestNetcon(unittest.TestCase):
    """
    def test_netcon(self):
        A = lattices.random_fin1DSimBTPS([["p0"],["p1"]])#,["p2"],["p3"]])
        a = A.to_tensor()
        k = a * a.conjugate()
        B = A.adjoint()
        C = A.tensors+A.bdts+B.tensors+B.bdts
        #print(C)
        l = contract_all_common(C)
        self.assertEqual(k,l)
        #print(k,l)
    """

class TestBrute(unittest.TestCase):
    def test_brute(self):
        A = TensorFrame((5,5),["i","j"], rpn=[0])
        B = TensorFrame((5,7,8),["j","k","l"], rpn=[1])
        C = TensorFrame((7,8,9),["k","l","m"], rpn=[2])
        hog = NetconBrute([A,B,C])
        res = hog.generate_troMemo()
        #for k,v in res.items():
        #    print(k,v)
        self.assertEqual(res[7]["cost"], 2745)
        f = hog.generate_contractor()
        self.assertEqual(f(3,5,7), 105)



if __name__=="__main__":
    unittest.main()