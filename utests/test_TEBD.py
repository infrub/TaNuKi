import unittest
import sys
sys.path.append('../')
from tanuki import *
from tanuki.onedim import *
from tanuki.matrices import *
import numpy as np
from math import sqrt
import copy




class TestRealFinTEBD(unittest.TestCase):
    def test01(self):
        def f(h,J): #Ising model
            H1 = Obc1DTMO(Tensor([[1,0],[0,-1]],["o","i"]),[["o"]],[["i"]],is_hermite=True)
            H2 = zeros_tensor((2,2,2,2),["o0","o1","i0","i1"])
            H2.data[0,0,0,0] = 1
            H2.data[0,1,0,1] = -1
            H2.data[1,0,1,0] = -1
            H2.data[1,1,1,1] = 1
            self.assertTrue(H2.is_hermite(["o0","o1"]))
            H2 = Obc1DTMO(H2,[["o0"],["o1"]],[["i0"],["i1"]],is_hermite=True)

            EH1 = H1.exp(-h).to_BTPO()
            EH2 = H2.exp(-J)
            self.assertTrue(EH2.tensor.is_hermite(["o0","o1"]))
            EH2 = EH2.to_BTPO()
            self.assertTrue(EH2.is_hermite)
            psi = random_fin1DBTPS([["p0"],["p1"],["p2"],["p3"]])

            for _ in range(100):
                psi.apply_everyplace([EH1,EH2], chi=4)

            return psi.to_TMS()

        a = f(1,0)
        self.assertTrue(abs(a.tensor.data[1,1,1,1]) > 1000*abs(a.tensor.data[0,1,1,0]))
        self.assertTrue(abs(a.tensor.data[1,1,1,1]) > 1000*abs(a.tensor.data[0,1,1,1]))
        self.assertTrue(abs(a.tensor.data[1,1,1,1]) > 1000*abs(a.tensor.data[0,1,0,1]))
        self.assertTrue(abs(a.tensor.data[1,1,1,1]) > 1000*abs(a.tensor.data[0,0,0,0]))
        self.assertAlmostEqual(a.tensor.norm(), 1)

        a = f(-6,0)
        self.assertTrue(abs(a.tensor.data[0,0,0,0]) > 1000*abs(a.tensor.data[0,1,1,0]))
        self.assertTrue(abs(a.tensor.data[0,0,0,0]) > 1000*abs(a.tensor.data[0,1,1,1]))
        self.assertTrue(abs(a.tensor.data[0,0,0,0]) > 1000*abs(a.tensor.data[0,1,0,1]))
        self.assertTrue(abs(a.tensor.data[0,0,0,0]) > 1000*abs(a.tensor.data[1,1,1,1]))
        self.assertAlmostEqual(a.tensor.norm(), 1)

        a = f(0,2)
        self.assertTrue(abs(a.tensor.data[0,1,0,1]) > 1000*abs(a.tensor.data[0,0,0,0]))
        self.assertTrue(abs(a.tensor.data[0,1,0,1]) > 1000*abs(a.tensor.data[0,1,1,0]))
        self.assertTrue(abs(a.tensor.data[0,1,0,1]) > 1000*abs(a.tensor.data[0,1,1,1]))
        self.assertTrue(abs(a.tensor.data[0,1,0,1]) > 1000*abs(a.tensor.data[1,1,1,1]))
        self.assertTrue(abs(a.tensor.data[1,0,1,0]) > 1000*abs(a.tensor.data[0,0,0,0]))
        self.assertTrue(abs(a.tensor.data[1,0,1,0]) > 1000*abs(a.tensor.data[0,1,1,0]))
        self.assertTrue(abs(a.tensor.data[1,0,1,0]) > 1000*abs(a.tensor.data[0,1,1,1]))
        self.assertTrue(abs(a.tensor.data[1,0,1,0]) > 1000*abs(a.tensor.data[1,1,1,1]))
        self.assertAlmostEqual(a.tensor.norm(), 1)



class TestImagFinTEBD(unittest.TestCase):
    def test01(self):
        J = 1
        psi = zeros_tensor((2,2,2,2),["p0","p1","p2","p3"])
        psi.data[0,0,0,0] = 0.2
        psi.data[0,1,0,1] = 1.0
        psi.data[1,0,1,0] = 1.0
        psi.data[1,1,1,1] = 1.4
        self.assertEqual(psi.norm(), 2.0)
        psi = Obc1DTMS(psi,[["p0"],["p1"],["p2"],["p3"]])
        psi = psi.to_BTPS()
        psi.canonize()
        self.assertAlmostEqual(psi.to_TMS().tensor.norm(),1.0)
        H2 = zeros_tensor((2,2,2,2),["o0","o1","i0","i1"])
        H2.data[0,0,0,0] = 1
        H2.data[0,1,0,1] = -1
        H2.data[1,0,1,0] = -1
        H2.data[1,1,1,1] = 1
        H2.data = J * H2.data
        E2high = 3*J
        E2low = -3*J
        self.assertTrue(H2.is_hermite(["o0","o1"]))
        H2 = Obc1DTMO(H2,[["o0"],["o1"]],[["i0"],["i1"]],is_hermite=True)
        T = 2 * np.pi / abs(E2high)
        Meh = 100
        dt = T / Meh
        EH2 = H2.exp(-1j * dt)
        self.assertTrue(EH2.tensor.is_unitary(["o0","o1"]))
        EH2 = EH2.to_BTPO()
        self.assertTrue(EH2.is_unitary)

        for _ in range(10):
            psi.apply_everyplace([EH2], chi=4)
        a = psi.to_TMS().tensor.round()
        self.assertNotEqual(a.data[0,0,0,0],0.1)

        for _ in range(Meh-10):
            psi.apply_everyplace([EH2], chi=4)
        a = psi.to_TMS().tensor.round()
        self.assertEqual(a.data[0,0,0,0],0.1)
        self.assertEqual(a.data[0,1,0,1],0.5)
        self.assertEqual(a.data[1,0,1,0],0.5)
        self.assertEqual(a.data[1,1,1,1],0.7)
        self.assertEqual(a.norm(),1.0)


if __name__=="__main__":
    unittest.main()
