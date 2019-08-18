import unittest
import sys
sys.path.append('../')
from tanuki import *
from tanuki.onedim import *
from tanuki.matrices import *
import numpy as np
from math import sqrt
import copy


#(Fin|Inf)1DSim(TM|TP|BTP)(S|O)


class TestITEBD(unittest.TestCase):
    def test_rtebd(self):
        def f(h,J): #Ising model
            H1 = Fin1DSimTMO(Tensor([[1,0],[0,-1]],["o","i"]),[["o"]],[["i"]],is_hermite=True)
            t = zeros_tensor((2,2,2,2),["o0","o1","i0","i1"])
            t.data[0,0,0,0] = 1
            t.data[0,1,0,1] = -1
            t.data[1,0,1,0] = -1
            t.data[1,1,1,1] = 1
            H2 = Fin1DSimTMO(t,[["o0"],["o1"]],[["i0"],["i1"]],is_hermite=True)

            EH1 = H1.exp(-h).to_BTPO()
            EH2 = H2.exp(-J).to_BTPO()
            psi = random_fin1DSimBTPS([["p0"],["p1"],["p2"],["p3"]])

            for _ in range(100):
                apply_everyplace_fin1DSimBTPS_fin1DSimBTPOs(psi, [EH1,EH2], chi=4)

            return psi.to_TMS()

        a = f(1,0)
        print(a)
        self.assertTrue(abs(a.tensor.data[1,1,1,1]) > 1000*abs(a.tensor.data[0,1,1,0]))
        self.assertTrue(abs(a.tensor.data[1,1,1,1]) > 1000*abs(a.tensor.data[0,1,1,1]))
        self.assertTrue(abs(a.tensor.data[1,1,1,1]) > 1000*abs(a.tensor.data[0,1,0,1]))
        self.assertTrue(abs(a.tensor.data[1,1,1,1]) > 1000*abs(a.tensor.data[0,0,0,0]))

        a = f(-6,0)
        print(a)
        self.assertTrue(abs(a.tensor.data[0,0,0,0]) > 1000*abs(a.tensor.data[0,1,1,0]))
        self.assertTrue(abs(a.tensor.data[0,0,0,0]) > 1000*abs(a.tensor.data[0,1,1,1]))
        self.assertTrue(abs(a.tensor.data[0,0,0,0]) > 1000*abs(a.tensor.data[0,1,0,1]))
        self.assertTrue(abs(a.tensor.data[0,0,0,0]) > 1000*abs(a.tensor.data[1,1,1,1]))

        a = f(0,2)
        print(a)
        self.assertTrue(abs(a.tensor.data[0,1,0,1]) > 1000*abs(a.tensor.data[0,0,0,0]))
        self.assertTrue(abs(a.tensor.data[0,1,0,1]) > 1000*abs(a.tensor.data[0,1,1,0]))
        self.assertTrue(abs(a.tensor.data[0,1,0,1]) > 1000*abs(a.tensor.data[0,1,1,1]))
        self.assertTrue(abs(a.tensor.data[0,1,0,1]) > 1000*abs(a.tensor.data[1,1,1,1]))
        self.assertTrue(abs(a.tensor.data[1,0,1,0]) > 1000*abs(a.tensor.data[0,0,0,0]))
        self.assertTrue(abs(a.tensor.data[1,0,1,0]) > 1000*abs(a.tensor.data[0,1,1,0]))
        self.assertTrue(abs(a.tensor.data[1,0,1,0]) > 1000*abs(a.tensor.data[0,1,1,1]))
        self.assertTrue(abs(a.tensor.data[1,0,1,0]) > 1000*abs(a.tensor.data[1,1,1,1]))



if __name__=="__main__":
    unittest.main()