import unittest
import sys
sys.path.append('../')
from tanuki import *
from tanuki.onedim import *
from tanuki.matrices import *
import numpy as np
from math import sqrt
import copy
import warnings




class TestBridgeBondEnv(unittest.TestCase):
    def test_fullrank(self):
        b = 10
        H_L = random_tensor((b,20),["kl","extractionl"])
        V_L = H_L * H_L.adjoint("kl",style="aster")
        H_R = random_tensor((b,20),["kr","extractionr"])
        V_R = H_R * H_R.adjoint("kr",style="aster")
        sigma0 = random_tensor((b,b),["kl","kr"])
        ENV = bondenv.BridgeBondEnv(V_L, V_R, ["kl"], ["kr"])

        Ms,Ss,Ns,er1s,er2s = [],[],[],[],[]
        def wa(chi):
            M,S,N = ENV.optimal_truncate(sigma0, chi=chi)
            er1 = ((M*S*N)-sigma0).norm()
            er2 = (((M*S*N)-sigma0)*H_L*H_R).norm()
            Ms.append(M)
            Ss.append(S)
            Ns.append(N)
            er1s.append(er1)
            er2s.append(er2)

        for i in range(1,b+1):
            wa(i)

        for i in range(b):
            self.assertEqual(Ss[i].dim(0), i+1)

        for i in range(b):
            for j in range(i+1):
                self.assertGreater(Ss[i].data[j], 0)
                if i < b-1:
                    self.assertEqual(Ss[i].data[j], Ss[i+1].data[j])
                if j < i:
                    self.assertGreater(Ss[i].data[j], Ss[i].data[j+1])

        for i in range(b-1):
            self.assertGreater(er2s[i],er2s[i+1])

        self.assertAlmostEqual(er1s[b-1],0)
        self.assertAlmostEqual(er2s[b-1],0)


    def test_degenerated(self):
        b = 10
        c = 8
        H_L = random_tensor((b,c),["kl","extractionl"])
        V_L = H_L * H_L.adjoint("kl",style="aster")
        H_R = random_tensor((b,c),["kr","extractionr"])
        V_R = H_R * H_R.adjoint("kr",style="aster")
        sigma0 = random_tensor((b,b),["kl","kr"])
        ENV = bondenv.BridgeBondEnv(V_L, V_R, ["kl"], ["kr"])

        Ms,Ss,Ns,er1s,er2s = [],[],[],[],[]
        def wa(chi):
            M,S,N = ENV.optimal_truncate(sigma0, chi=chi)
            er1 = ((M*S*N)-sigma0).norm()
            er2 = (((M*S*N)-sigma0)*H_L*H_R).norm()
            Ms.append(M)
            Ss.append(S)
            Ns.append(N)
            er1s.append(er1)
            er2s.append(er2)

        for i in range(1,b+1):
            wa(i)

        for i in range(b):
            self.assertEqual(Ss[i].dim(0), min(i+1,c))

        for i in range(b):
            for j in range(min(i+1,c)):
                self.assertGreaterEqual(Ss[i].data[j], 0)
                if i < b-1:
                    self.assertEqual(Ss[i].data[j], Ss[i+1].data[j])
                if j < min(i+1,c)-1:
                    self.assertGreaterEqual(Ss[i].data[j], Ss[i].data[j+1])

        for i in range(b-1):
            self.assertGreaterEqual(er1s[i],er1s[i+1])
            self.assertGreaterEqual(er2s[i],er2s[i+1])

        self.assertAlmostEqual(er2s[c],0)
        self.assertAlmostEqual(er2s[b-1],0)



class TestUnbridgeBondEnv(unittest.TestCase):
    def test_fullrank(self):
        b = 10
        H = random_tensor((b,b,b*b+5),["kl","kr","extraction"])
        V = H * H.adjoint(["kl","kr"],style="aster")
        sigma0 = random_tensor((b,b),["kl","kr"])
        ENV = bondenv.UnbridgeBondEnv(V, ["kl"], ["kr"])

        Ms,Ss,Ns,er1s,er2s = [],[],[],[],[]
        def wa(chi):
            M,S,N = ENV.optimal_truncate(sigma0, chi=chi)
            er1 = ((M*S*N)-sigma0).norm()
            er2 = (((M*S*N)-sigma0)*H).norm()
            Ms.append(M)
            Ss.append(S)
            Ns.append(N)
            er1s.append(er1)
            er2s.append(er2)

        for i in range(1,b+1):
            wa(i)

        for i in range(b):
            self.assertEqual(Ss[i].dim(0), i+1)

        for i in range(b):
            for j in range(i+1):
                self.assertGreaterEqual(Ss[i].data[j], 0)
                if j < i:
                    self.assertGreaterEqual(Ss[i].data[j], Ss[i].data[j+1])

        for i in range(b-1):
            self.assertGreater(er2s[i],er2s[i+1])

        self.assertAlmostEqual(er1s[b-1],0)
        self.assertAlmostEqual(er2s[b-1],0)


    def test_degenerated(self):
        b = 10
        nazo = 8 #TODO why??
        c = 6 #TODO why??
        H = random_tensor((b,b,nazo*nazo),["kl","kr","extraction"])
        V = H * H.adjoint(["kl","kr"],style="aster")
        sigma0 = random_tensor((b,b),["kl","kr"])
        ENV = bondenv.UnbridgeBondEnv(V, ["kl"], ["kr"])

        Ms,Ss,Ns,er1s,er2s = [],[],[],[],[]
        def wa(chi):
            M,S,N = ENV.optimal_truncate(sigma0, chi=chi)
            er1 = ((M*S*N)-sigma0).norm()
            er2 = (((M*S*N)-sigma0)*H).norm()
            Ms.append(M)
            Ss.append(S)
            Ns.append(N)
            er1s.append(er1)
            er2s.append(er2)

        for i in range(1,b+1):
            wa(i)

        for i in range(b):
            self.assertEqual(Ss[i].dim(0), min(i+1,c))

        for i in range(b):
            for j in range(min(i+1,c)):
                self.assertGreater(Ss[i].data[j], 0)
                if j < min(i+1,c)-1:
                    self.assertGreater(Ss[i].data[j], Ss[i].data[j+1])

        for i in range(b-1):
            self.assertTrue(er2s[i] > er2s[i+1] or er2s[i+1] < 1e-10)

        self.assertAlmostEqual(er2s[c],0)
        self.assertAlmostEqual(er2s[b-1],0)

        

if __name__ == "__main__":
    unittest.main()
