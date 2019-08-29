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

        n = min(n,b**2)
        max_chi_can_use_iterating_method = n // b # = floor(n/b)
        min_chi_can_use_exact_solving = (n-1)//b+1 # = ceil(n/b)

        Ms,Ss,Ns,er1s,er2s,memos = [None],[None],[None],[None],[None],[None]
        for c in range(1,mc+3):
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

            self.assertEqual(memo["chi"], min(c,mc))
            if memo["used_algorithm"] == "exact_solving":
                self.assertGreaterEqual(memo["chi"], min_chi_can_use_exact_solving)
                self.assertTrue(memo["has_enough_degree_of_freedom_to_solve_exactly"])
                self.assertAlmostEqual(er2,0, places=12)
            else:
                self.assertLessEqual(memo["chi"], max_chi_can_use_iterating_method)

            # S is positive and descending
            for j in range(memo["chi"]):
                self.assertGreaterEqual(Ss[c].data[j], 0)
                if j < memo["chi"]-1:
                    self.assertGreaterEqual(Ss[c].data[j], Ss[c].data[j+1])

        for c in range(1,mc+2):
            self.assertTrue(er2s[c] >= er2s[c+1] or er2s[c+1] < 1e-8)
    
    def test_a0e1(self):
        # [1<=n<b]
        # max_chi_can_use_iterating_method = 0
        # min_chi_can_use_exact_solving = 1
        # exact_solve(1)
        b = 10
        mc = 1
        for n in [1,5,9]:
            self.test_iroiro(b,n,mc)
    
    def test_a1e1(self):
        # [n==b]
        # max_chi_can_use_iterating_method = 1
        # min_chi_can_use_exact_solving = 1
        b = 10
        mc = 1
        n = 10
        self.test_iroiro(b,n,mc)

    def test_a1e2(self):
        # [b<=n<2*b]
        # max_chi_can_use_iterating_method = 1
        # min_chi_can_use_exact_solving = 2
        b = 10
        mc = 2
        for n in [11,15,19]:
            self.test_iroiro(b,n,mc)
    
    def test_hanpa(self):
        # [2b<=n<b**2]
        b = 10
        for n,mc in [(20,2), (30,3), (51,6), (96,10), (99,10)]:
            self.test_iroiro(b,n,mc)

    def test_fullrank(self):
        # [n==b**2] (if input n>b**2, it becomes n=b**2)
        # max_chi_can_use_iterating_method = b
        # min_chi_can_use_exact_solving = b
        b = 10
        mc = b
        for n in [100, 102, 104]:
            self.test_iroiro(b,n,mc)
    


if __name__ == "__main__":
    unittest.main()
