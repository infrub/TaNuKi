import unittest
import sys
sys.path.append('../')
from tanuki import *
from tanuki.lattices import *
import numpy as np
from math import sqrt


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




class TestInf1DSimBTPS(unittest.TestCase):
    def test_instant(self):
        phys_labelss = [["p0"], ["p1"], ["p2"]]
        A = random_inf1DSimBTPS(phys_labelss)
        a = A.to_tensor()
        self.assertTrue(eq_list(a.labels, ["p0","p1","p2"]))

    def test_transfer_eigen(self):
        A = random_inf1DSimBTPS([["p0"], ["p10","p11"], ["p2"]], virt_labelss=[["v0"],["v10","v11"],["v2"]], chi=4)
        self.assertTrue(eq_list(A.get_guessed_phys_labels_site(1), ["p10","p11"]))
        w_L, V_L = A.get_left_transfer_eigen()
        w_R, V_R = A.get_right_transfer_eigen()
        self.assertAlmostEqual(w_L, w_R, 10)

    def test_canonize_end(self):
        A = random_inf1DSimBTPS([["p0"], ["p10","p11"], ["p2"]], virt_labelss=[["v0"],["v1"],["v2"]])
        a1 = A.to_tensor()
        w_L, V_L = A.get_left_transfer_eigen()
        w_R, V_R = A.get_right_transfer_eigen()
        self.assertFalse(V_L.is_prop_identity(A.get_ket_left_labels_bond(0)))
        self.assertFalse(V_R.is_prop_identity(A.get_ket_right_labels_bond(0)))
        A.canonize_end(transfer_normalize=False)
        a2 = A.to_tensor()
        w_L, V_L = A.get_left_transfer_eigen()
        w_R, V_R = A.get_right_transfer_eigen()
        self.assertTrue(V_L.is_prop_identity(A.get_ket_left_labels_bond(0)))
        self.assertTrue(V_R.is_prop_identity(A.get_ket_right_labels_bond(0)))
        self.assertEqual(a2, a1)

        A = random_inf1DSimBTPS([["p0"], ["p1"], ["p2"]], virt_labelss=[["v0"],["v10","v11"],["v2"]])
        a1 = A.to_tensor()
        w_L, V_L = A.get_left_transfer_eigen()
        w_R, V_R = A.get_right_transfer_eigen()
        w_L1 = w_L
        self.assertFalse(V_L.is_prop_identity(A.get_ket_left_labels_bond(0)))
        self.assertFalse(V_R.is_prop_identity(A.get_ket_right_labels_bond(0)))
        self.assertNotAlmostEqual(w_L, 1, 3)
        self.assertNotAlmostEqual(w_R, 1, 3)
        A.canonize_end(transfer_normalize=True)
        a2 = A.to_tensor()
        w_L, V_L = A.get_left_transfer_eigen()
        w_R, V_R = A.get_right_transfer_eigen()
        self.assertTrue(V_L.is_prop_identity(A.get_ket_left_labels_bond(0)))
        self.assertTrue(V_R.is_prop_identity(A.get_ket_right_labels_bond(0)))
        self.assertAlmostEqual(w_L, 1)
        self.assertAlmostEqual(w_R, 1)
        self.assertEqual(a2*sqrt(w_L1), a1)

    def test_canonize(self):
        A = random_inf1DSimBTPS([["p0"], ["p1"], ["p2"]], virt_labelss=[["v0"],["v1"],["v2"]])
        a1 = A.to_tensor()
        w_L1, V_L1 = A.get_left_transfer_eigen()
        A.canonize(transfer_normalize=True)
        a2 = A.to_tensor()
        self.assertEqual(a2*sqrt(w_L1), a1)
        re = A.is_canonical()
        self.assertTrue(re)
        for lr in ["left","right"]:
            for i in range(len(A)):
                self.assertTrue(re[lr][i])
                self.assertAlmostEqual(re[lr][i]["factor"], 1, 10)

    def test_canonize_end_not_zero(self):
        A = random_inf1DSimBTPS([["p0"], ["p10","p11"], ["p2"]], virt_labelss=[["v0"],["v1"],["v2"]])
        a1 = A.to_tensor()
        w_L, V_L = A.get_left_transfer_eigen(4)
        w_R, V_R = A.get_right_transfer_eigen(2)
        w_L1 = w_L
        self.assertNotAlmostEqual(w_L, 1, 3)
        self.assertNotAlmostEqual(w_R, 1, 3)
        A.canonize_end(4, transfer_normalize=False)
        a2 = A.to_tensor()
        self.assertEqual(a2, a1)

        A = random_inf1DSimBTPS([["p0"], ["p1"], ["p2"]], virt_labelss=[["v0"],["v10","v11"],["v2"]])
        a1 = A.to_tensor()
        w_L, V_L = A.get_left_transfer_eigen(4)
        w_R, V_R = A.get_right_transfer_eigen(3)
        w_L1 = w_L
        self.assertNotAlmostEqual(w_L, 1, 3)
        self.assertNotAlmostEqual(w_R, 1, 3)
        A.canonize_end(1, transfer_normalize=True)
        a2 = A.to_tensor()
        w_L, V_L = A.get_left_transfer_eigen(9)
        w_R, V_R = A.get_right_transfer_eigen(-5)
        self.assertAlmostEqual(w_L, 1)
        self.assertAlmostEqual(w_R, 1)
        self.assertEqual(a2*sqrt(w_L1), a1)


class TestHeritation(unittest.TestCase):
    def test_inherit(self):
        A = random_inf1DSimBTPS([["p0"], ["p1"], ["p2"]], virt_labelss=[["v0"],["v1"],["v2"]])







if __name__=="__main__":
    unittest.main()