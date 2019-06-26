import unittest
import sys
sys.path.append('../')
from tanuki import *
from tanuki.lattices import *
import numpy as np
from math import sqrt

"""
class TestFin1DSimTPS(unittest.TestCase):
    def test_nank(self):
        A = random_tensor((2,3),["p0","v0"])
        B = random_tensor((3,2,3),["v0","p1","v1"])
        C = random_tensor((3,2),["v1","p2"])
        S = Fin1DSimTPS([A,B,C])
        s = S.to_tensor()

        T = S.to_BTPS()
        T.both_canonize(end_dealing="no")
        t = T.to_tensor()

        self.assertEqual(s,t)

        print(T)
        self.assertTrue(T.is_both_canonical(end_dealing="no"))
"""


class TestFin1DSimBTPS(unittest.TestCase):
    def test_instant(self):
        phys_labelss = [["p0"], ["p1"], ["p2"]]
        A = random_fin1DSimBTPS(phys_labelss)
        a = A.to_tensor()
        self.assertTrue(eq_list(a.labels, ["p0","p1","p2"]))
        A = random_fin1DSimBTPS([["p0"], ["p10","p11"], ["p2"], ["p3"]], virt_labelss=[["v0"],["v10,v11"],["v2"]], chi=4)
        self.assertTrue(eq_list(A.get_guessed_phys_labels_site(1), ["p10","p11"]))
        self.assertEqual(A.tensors[1].size, 64)
        a = A.to_tensor()
        self.assertTrue(eq_list(a.labels, ["p0","p10","p11","p2","p3"]))

    def test_canonize_site(self):
        phys_labelss = [["p0"], ["p1"], ["p2"]]
        A = random_fin1DSimBTPS(phys_labelss)
        self.assertFalse(A.is_left_canonical_site(0))
        A.left_canonize_site(0)
        self.assertTrue(A.is_left_canonical_site(0))
        A.left_canonize(1,3)
        self.assertTrue(A.is_left_canonical_site(0))
        self.assertTrue(A.is_left_canonical_site(1))
        self.assertTrue(A.is_left_canonical_site(2))

    def test_both_canonize(self):
        phys_labelss = [["p0"], ["p1"], ["p2"]]
        A = random_fin1DSimBTPS(phys_labelss)
        a1 = A.to_tensor()
        self.assertFalse(A.is_both_canonical())
        A.both_canonize(end_dealing="no")
        a2 = A.to_tensor()
        self.assertTrue(A.is_both_canonical())
        self.assertEqual(a2, a1)
        A.both_canonize(end_dealing="normalize")
        a3 = A.to_tensor()
        self.assertTrue(A.is_both_canonical())
        self.assertEqual(a3, a1.normalize())
        self.assertAlmostEqual(a3.norm(), 1, 10)




if __name__=="__main__":
    unittest.main()