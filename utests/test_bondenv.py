import unittest
import sys
sys.path.append('../')
from tanuki import *
from tanuki.onedim import *
from tanuki.matrices import *
from math import sqrt
import copy
import warnings
import faulthandler
faulthandler.enable()




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
    def test_iroiro(self, b=None, n=None, mc=None): # mc is yosou sareru hitstop of chi
        if b is None: return

        H = random_tensor((b,b,n),["kl","kr","extraction"])
        V = H * H.adjoint(["kl","kr"],style="aster")
        sigma0 = random_tensor((b,b),["kl","kr"])
        ENV = bondenv.UnbridgeBondEnv(V, ["kl"], ["kr"])

        Ms,Ss,Ns,er1s,er2s,memos = [None],[None],[None],[None],[None],[None]
        def wa(c):
            memo = {}
            M,S,N = ENV.optimal_truncate(sigma0, chi=c, memo=memo)
            er1 = ((M*S*N)-sigma0).norm()
            er2 = (((M*S*N)-sigma0)*H).norm()
            Ms.append(M)
            Ss.append(S)
            Ns.append(N)
            er1s.append(er1)
            er2s.append(er2)
            memos.append(memo)
            print(b,n,c,er2,memo)

        for c in range(1,mc+3):
            wa(c)

        for c in range(1,mc+3): # S is positive and descending
            for j in range(memos[c]["chi"]):
                self.assertGreaterEqual(Ss[c].data[j], 0)
                if j < memos[c]["chi"]-1:
                    self.assertGreaterEqual(Ss[c].data[j], Ss[c].data[j+1])

        for c in range(1,mc+3):
            self.assertEqual(memos[c]["chi"], min(c,mc))
            if memos[c]["exactly_solvable"]:
                self.assertAlmostEqual(er2s[c],0, places=7)
            else:
                self.assertEqual(memos[c]["used_algorithm"], "iterating_method")

        for c in range(1,mc+2):
            self.assertTrue(er2s[c] >= er2s[c+1] or er2s[c+1] < 1e-8)

    def test_a0e1(self):
        # [1<=n<b]
        # avoiding_singular_chi == 0
        # exactly_solvable_chi == 1
        # exact_solve(1)
        b = 10
        mc = 1
        for n in [1,5,9]:
            self.test_iroiro(b,n,mc)

    def test_a1e1(self):
        # [b<=n<1.5b]
        # avoiding_singular_chi == 1
        # exactly_solvable_chi == 1
        # exact_solve(1)
        b = 10
        mc = 1
        for n in [10,12,14]:
            self.test_iroiro(b,n,mc)

    def test_a1e2(self):
        # [1.5b<=n<2b]
        # avoiding_singular_chi == 1
        # exactly_solvable_chi == 2
        # exact_solve(2)
        b = 10
        mc = 2
        for n in [15,17,19]:
            self.test_iroiro(b,n,mc)

    def test_hanpa(self):
        # [2b<=n<b**2-1]
        # exactly_solvable_chi <= avoiding_singular_chi
        # obviously no problem
        # mc = exactly_solvable_chi
        b = 10
        for n,mc in [(20,2), (30,2), (48,4), (90,8), (98,9)]:
            #           0,      1,      0,      1,     0      == avoiding_singular_chi - exactly_solvable_chi
            self.test_iroiro(b,n,mc)

    def test_abm1eb(self):
        # [n==b**2-1]
        # avoiding_singular_chi == b-1
        # exactly_solvable_chi == b
        # exact_solve(b)
        b = 10
        n = b**2-1
        mc = b
        self.test_iroiro(b,n,mc)

    def test_fullrank(self):
        # [n==b**2] (if input n>b**2, it becomes n=b**2)
        # avoiding_singular_chi == b
        # exactly_solvable_chi == b
        # exact_solve(b)
        b = 10
        mc = b
        for n in [100, 102, 104]:
            self.test_iroiro(b,n,mc)



if __name__ == "__main__":
    unittest.main()
