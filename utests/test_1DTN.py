import unittest
import sys
sys.path.append('../')
from tanuki import *
from tanuki.lattices import *
import numpy as np
from math import sqrt

class TestFin1DSimTPS(unittest.TestCase):
    def test_nank(self):
        A = random_tensor((2,3),["p0","v0"])
        B = random_tensor((3,2,3),["v0","p1","v1"])
        C = random_tensor((3,2),["v1","p2"])
        S = Fin1DSimTPS([A,B,C])
        print(S)
        s = S.to_tensor()
        print(s)

        T = S.to_BTPS()
        print(T)
        T.both_canonize(end_dealing="no")
        print(T)
        t = T.to_tensor()
        print(t)
        self.assertEqual(s,t)

        print(T)
        self.assertTrue(T.is_both_canonical(end_dealing="no"))



if __name__=="__main__":
    unittest.main()