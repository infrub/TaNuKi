import unittest
import sys
sys.path.append('../')
from tanuki import *
from tanuki.onedim import *
import numpy as np
from math import sqrt
import copy


#(Fin|Inf)1DSim(TM|TP|BTP)(S|O)


class TestObc1DBTPS(unittest.TestCase):
    def test_instant(self):
        phys_labelss = [["p0"], ["p1"], ["p2"]]
        A = random_fin1DBTPS(phys_labelss)
        a = A.to_tensor()
        self.assertTrue(eq_list(a.labels, ["p0","p1","p2"]))
        A = random_fin1DBTPS([["p0"], ["p10","p11"], ["p2"], ["p3"]], virt_labelss=[["v0"],["v10,v11"],["v2"]], chi=4)
        self.assertTrue(eq_list(A.get_guessed_phys_labels_site(1), ["p10","p11"]))
        self.assertEqual(A.tensors[1].size, 64)
        a = A.to_tensor()
        self.assertTrue(eq_list(a.labels, ["p0","p10","p11","p2","p3"]))

    def test_canonize_site(self):
        phys_labelss = [["p0"], ["p1"], ["p2"]]
        A = random_fin1DBTPS(phys_labelss)
        self.assertFalse(A.is_locally_left_canonical_around_bond(1))
        A.locally_left_canonize_around_bond(1)
        self.assertTrue(A.is_locally_left_canonical_around_bond(1))
        A.globally_left_canonize_upto(3,1)
        self.assertTrue(A.is_locally_left_canonical_around_bond(1))
        self.assertTrue(A.is_locally_left_canonical_around_bond(2))
        self.assertTrue(A.is_locally_left_canonical_around_bond(3))

    def test_universally_canonize(self):
        phys_labelss = [["p0"], ["p1"], ["p2"]]
        A = random_fin1DBTPS(phys_labelss)
        a1 = A.to_tensor()
        self.assertFalse(A.is_canonical())
        A.universally_canonize(end_dealing="no")
        a2 = A.to_tensor()
        self.assertTrue(A.is_canonical())
        self.assertEqual(a2, a1)
        A.universally_canonize(end_dealing="normalize")
        a3 = A.to_tensor()
        self.assertTrue(A.is_canonical())
        self.assertEqual(a3, a1.normalize())
        self.assertAlmostEqual(a3.norm(), 1, 10)

    def test_convert(self):
        phys_labelss = [["p0"], ["p1"], ["p2"]]
        A = random_fin1DTPS(phys_labelss)
        self.assertEqual(A.to_BTPS().to_tensor(), A.to_tensor())
        B = random_fin1DBTPS(phys_labelss)
        self.assertEqual(B.to_TPS().to_tensor(), B.to_tensor())




class TestInf1DBTPS(unittest.TestCase):
    def test_instant(self):
        phys_labelss = [["p0"], ["p1"], ["p2"]]
        A = random_inf1DBTPS(phys_labelss)
        a = A.to_tensor()
        self.assertTrue(eq_list(a.labels, ["p0","p1","p2"]))

    def test_transfer_eigen(self):
        A = random_inf1DBTPS([["p0"], ["p10","p11"], ["p2"]], virt_labelss=[["v0"],["v10","v11"],["v2"]], chi=4)
        self.assertTrue(eq_list(A.get_guessed_phys_labels_site(1), ["p10","p11"]))
        w_L, V_L = A.get_left_transfer_eigen()
        w_R, V_R = A.get_right_transfer_eigen()
        self.assertTrue( abs(w_L-w_R) < 1e-5*abs(w_L) )

    def test_universally_canonize_around_end_bond(self):
        A = random_inf1DBTPS([["p0"], ["p10","p11"], ["p2"]], virt_labelss=[["v0"],["v1"],["v2"]])
        a1 = A.to_tensor()
        w_L, V_L = A.get_left_transfer_eigen()
        w_R, V_R = A.get_right_transfer_eigen()
        self.assertFalse(V_L.is_prop_identity(A.get_ket_left_labels_bond(0)))
        self.assertFalse(V_R.is_prop_identity(A.get_ket_right_labels_bond(0)))
        A.universally_canonize_around_end_bond(transfer_normalize=False)
        a2 = A.to_tensor()
        w_L, V_L = A.get_left_transfer_eigen()
        w_R, V_R = A.get_right_transfer_eigen()
        self.assertTrue(V_L.is_prop_identity(A.get_ket_left_labels_bond(0), check_rtol=1e-4, check_atol=1e-4))
        self.assertTrue(V_R.is_prop_identity(A.get_ket_right_labels_bond(0), check_rtol=1e-4, check_atol=1e-4))
        self.assertEqual(a2, a1)

        A = random_inf1DBTPS([["p0"], ["p1"], ["p2"]], virt_labelss=[["v0"],["v10","v11"],["v2"]])
        a1 = A.to_tensor()
        w_L, V_L = A.get_left_transfer_eigen()
        w_R, V_R = A.get_right_transfer_eigen()
        w_L1 = w_L
        self.assertFalse(V_L.is_prop_identity(A.get_ket_left_labels_bond(0)))
        self.assertFalse(V_R.is_prop_identity(A.get_ket_right_labels_bond(0)))
        self.assertNotAlmostEqual(w_L, 1, 3)
        self.assertNotAlmostEqual(w_R, 1, 3)
        A.universally_canonize_around_end_bond(transfer_normalize=True)
        a2 = A.to_tensor()
        w_L, V_L = A.get_left_transfer_eigen()
        w_R, V_R = A.get_right_transfer_eigen()
        self.assertTrue(V_L.is_prop_identity(A.get_ket_left_labels_bond(0), check_rtol=1e-4, check_atol=1e-4))
        self.assertTrue(V_R.is_prop_identity(A.get_ket_right_labels_bond(0), check_rtol=1e-4, check_atol=1e-4))
        self.assertAlmostEqual(w_L, 1, 5)
        self.assertAlmostEqual(w_R, 1, 5)
        self.assertEqual(a2*sqrt(w_L1), a1)

    def test_canonize(self):
        A = random_inf1DBTPS([["p0"], ["p1"], ["p2"]], virt_labelss=[["v0"],["v1"],["v2"]])
        a1 = A.to_tensor()
        w_L1, V_L1 = A.get_left_transfer_eigen()
        A.canonize(transfer_normalize=True)
        a2 = A.to_tensor()
        self.assertEqual(a2*sqrt(w_L1), a1)
        re = A.is_canonical(check_rtol=1e-4,check_atol=1e-6)
        self.assertTrue(re)
        for lr in ["left","right"]:
            for i in range(len(A)):
                self.assertTrue(re[lr][i])
                self.assertAlmostEqual(re[lr][i]["factor"], 1, 5)

    def test_universally_canonize_around_end_bond_not_zero(self):
        A = random_inf1DBTPS([["p0"], ["p10","p11"], ["p2"]], virt_labelss=[["v0"],["v1"],["v2"]])
        a1 = A.to_tensor()
        w_L, V_L = A.get_left_transfer_eigen(4)
        w_R, V_R = A.get_right_transfer_eigen(2)
        w_L1 = w_L
        self.assertNotAlmostEqual(w_L, 1, 3)
        self.assertNotAlmostEqual(w_R, 1, 3)
        A.universally_canonize_around_end_bond(4, transfer_normalize=False)
        a2 = A.to_tensor()
        self.assertEqual(a2, a1)

        A = random_inf1DBTPS([["p0"], ["p1"], ["p2"]], virt_labelss=[["v0"],["v10","v11"],["v2"]])
        a1 = A.to_tensor()
        w_L, V_L = A.get_left_transfer_eigen(4)
        w_R, V_R = A.get_right_transfer_eigen(3)
        w_L1 = w_L
        self.assertNotAlmostEqual(w_L, 1, 3)
        self.assertNotAlmostEqual(w_R, 1, 3)
        A.universally_canonize_around_end_bond(1, transfer_normalize=True)
        a2 = A.to_tensor()
        w_L, V_L = A.get_left_transfer_eigen(9)
        w_R, V_R = A.get_right_transfer_eigen(-5)
        self.assertAlmostEqual(w_L, 1, 5)
        self.assertAlmostEqual(w_R, 1, 5)
        self.assertEqual(a2*sqrt(w_L1), a1)




class TestMPO(unittest.TestCase):
    def test_01(self):
        A = zeros_tensor((2,2,4), ["in0", "out0", "w"])
        B = zeros_tensor((2,2,4), ["in1", "out1", "w"])
        A.data[0,0,0], B.data[0,0,0] = 1.0, 1.0
        A.data[0,1,1], B.data[1,0,1] = 1.0, 1.0
        A.data[1,0,2], B.data[0,1,2] = 1.0, 1.0
        A.data[1,1,3], B.data[1,1,3] = 1.0, 1.0
        mpo = Obc1DTPO([A,B], physin_labelss=[["in0"],["in1"]], physout_labelss=[["out0"],["out1"]], is_unitary=True)
        self.assertTrue(mpo.to_tensor().is_unitary(["out0", "out1"]))
        bmpo = mpo.to_BTPO()
        self.assertEqual(mpo.to_tensor(), bmpo.to_tensor())

        mps = random_fin1DBTPS([["p0"], ["p1"], ["p2"]])
        mps.canonize()
        mps0 = copyModule.deepcopy(mps)
        self.assertAlmostEqual(mps.to_tensor().norm(), 1)
        self.assertTrue(mps.is_canonical())
        apply_fin1DSimBTPS_fin1DSimBTPO(mps, bmpo)
        self.assertTrue(mps.is_canonical())
        self.assertFalse(mps.to_tensor().__eq__(mps0.to_tensor(), skipLabelSort=True))
        apply_fin1DSimBTPS_fin1DSimBTPO(mps, bmpo)
        self.assertTrue(mps.is_canonical())
        self.assertTrue(mps.to_tensor().__eq__(mps0.to_tensor(), skipLabelSort=True))

    def test_02(self):
        G, _, _ = tensor_svd(random_tensor((27, 27), ["a", "b"], dtype=complex), ["a"])
        G.labels = ["a","b"]
        G = G.split_index("a", (3,3,3), ["out0", "out1", "out2"])
        G = G.split_index("b", (3,3,3), ["in0", "in1", "in2"])
        self.assertTrue(G.is_unitary(["out0","out1","out2"]))
        a0, s1, G = tensor_svd(G, ["out0", "in0"])
        a1, s2, a2 = tensor_svd(G, ["out1", "in1"])
        mpo = Obc1DBTPO([a0,a1,a2],[s1,s2],physin_labelss=[["in0"],["in1"],["in2"]],physout_labelss=[["out0"],["out1"],["out2"]], is_unitary=True)

        mps0 = random_fin1DBTPS([["p0"], ["p1"], ["p2"], ["p3"], ["p4"], ["p5"]], chi=5, phys_dim=3)
        mps = copyModule.deepcopy(mps0)
        self.assertFalse(mps.is_canonical())
        apply_fin1DSimBTPS_fin1DSimBTPO(mps, mpo, 0, keep_phys_labels=True)
        apply_fin1DSimBTPS_fin1DSimBTPO(mps, mpo, 1, keep_phys_labels=True)
        apply_fin1DSimBTPS_fin1DSimBTPO(mps, mpo, 2, keep_phys_labels=True)
        self.assertFalse(mps.is_canonical())
        apply_fin1DSimBTPS_fin1DSimBTPO(mps, mpo, 3, keep_phys_labels=True)
        apply_fin1DSimBTPS_fin1DSimBTPO(mps, mpo, 2, keep_phys_labels=True)
        apply_fin1DSimBTPS_fin1DSimBTPO(mps, mpo, 1, keep_phys_labels=True)
        self.assertTrue(mps.is_canonical())

    def test_03(self):
        G, _, _ = tensor_svd(random_tensor((27, 27), ["a", "b"], dtype=complex), ["a"])
        G.labels = ["a","b"]
        G = G.split_index("a", (3,3,3), ["out0", "out1", "out2"])
        G = G.split_index("b", (3,3,3), ["in0", "in1", "in2"])
        self.assertTrue(G.is_unitary(["out0","out1","out2"]))
        a0, s1, G = tensor_svd(G, ["out0", "in0"])
        a1, s2, a2 = tensor_svd(G, ["out1", "in1"])
        mpo = Obc1DBTPO([a0,a1,a2],[s1,s2],physin_labelss=[["in0"],["in1"],["in2"]],physout_labelss=[["out0"],["out1"],["out2"]], is_unitary=True)
        mps0 = random_fin1DBTPS([["p0"], ["p1"], ["p2"], ["p3"], ["p4"], ["p5"]], chi=5, phys_dim=3)

        mps = copyModule.deepcopy(mps0)
        mps.canonize()
        apply_fin1DSimBTPS_fin1DSimBTPO(mps, mpo, 0, keep_phys_labels=True)
        apply_fin1DSimBTPS_fin1DSimBTPO(mps, mpo, 1, keep_phys_labels=True)
        apply_fin1DSimBTPS_fin1DSimBTPO(mps, mpo, 2, keep_phys_labels=True)
        apply_fin1DSimBTPS_fin1DSimBTPO(mps, mpo, 3, keep_phys_labels=True)
        apply_fin1DSimBTPS_fin1DSimBTPO(mps, mpo, 2, keep_phys_labels=True)
        apply_fin1DSimBTPS_fin1DSimBTPO(mps, mpo, 1, keep_phys_labels=True)
        self.assertTrue(mps.is_canonical())
        mps1 = mps

        mps = copyModule.deepcopy(mps0)
        mps.canonize()
        apply_fin1DSimBTPS_fin1DSimBTPO(mps, mpo, 0, chi=26, keep_phys_labels=True)
        apply_fin1DSimBTPS_fin1DSimBTPO(mps, mpo, 1, chi=26, keep_phys_labels=True)
        apply_fin1DSimBTPS_fin1DSimBTPO(mps, mpo, 2, chi=26, keep_phys_labels=True)
        apply_fin1DSimBTPS_fin1DSimBTPO(mps, mpo, 3, chi=26, keep_phys_labels=True)
        apply_fin1DSimBTPS_fin1DSimBTPO(mps, mpo, 2, chi=26, keep_phys_labels=True)
        apply_fin1DSimBTPS_fin1DSimBTPO(mps, mpo, 1, chi=26, keep_phys_labels=True)
        mps2 = mps

        a = mps1.to_tensor()
        ah = mps1.adjoint().to_tensor()
        b = mps2.to_tensor()
        bh = mps2.adjoint().to_tensor()

        diff1 = (a - b).norm()**2
        diff2 = ((ah*a) - (ah*b) - (bh*a) + (bh*b)).real().to_scalar()
        self.assertAlmostEqual((ah*a).to_scalar(), 1.0)
        self.assertAlmostEqual((bh*a).to_scalar(), 1.0, 3)
        self.assertAlmostEqual((ah*b).to_scalar(), 1.0, 3)
        self.assertAlmostEqual((bh*b).to_scalar(), 1.0, 2)
        self.assertLess(diff1, 3e-4)
        self.assertLess(diff2, 3e-4)
        self.assertAlmostEqual(diff1, diff2)


    def test_04(self):
        G = zeros_tensor((2,2,2,2), ["out0","out1","in0","in1"])
        G.data[0,0,0,0] = 1.5
        G.data[0,1,0,1] = -1.5
        G.data[1,0,1,0] = -1.5
        G.data[1,1,1,1] = 1.5
        G = Obc1DTMO(G, [["out0"],["out1"]],[["in0"],["in1"]])
        G = G.exp(-2j)
        F = zeros_tensor((2,2,2,2), ["out0","out1","in0","in1"])
        F.data[0,0,0,0] = np.exp(-3j)
        F.data[0,1,0,1] = np.exp(3j)
        F.data[1,0,1,0] = np.exp(3j)
        F.data[1,1,1,1] = np.exp(-3j)
        self.assertEqual(G.tensor, F)
        self.assertTrue(G.tensor.is_unitary(["out0","out1"]))
        G = G.to_BTPO()
        self.assertEqual(len(G),2)





if __name__=="__main__":
    unittest.main()