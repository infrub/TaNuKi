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
    def test_fullrank(self):
        b = 8
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
        b = 8
        n = 48
        c = 6
        H = random_tensor((b,b,n),["kl","kr","extraction"])
        V = H * H.adjoint(["kl","kr"],style="aster")
        sigma0 = random_tensor((b,b),["kl","kr"])
        ENV = bondenv.UnbridgeBondEnv(V, ["kl"], ["kr"])

        Ms,Ss,Ns,er1s,er2s = [],[],[],[],[]
        def wa(chi):
            memo = {}
            M,S,N = ENV.optimal_truncate(sigma0, chi=chi, memo=memo)
            self.assertFalse(memo["env_is_crazy_degenerated"])
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
            self.assertTrue(er2s[i] > er2s[i+1] or er2s[i+1] < 1e-8)

        self.assertAlmostEqual(er2s[c],0)
        self.assertAlmostEqual(er2s[b-1],0)

    
    def test_crazy_degenerated(self):
        b = 8
        n = 7
        c = 1
        H = random_tensor((b,b,n),["kl","kr","extraction"])
        V = H * H.adjoint(["kl","kr"],style="aster")
        V.is_hermite(["kl","kr"]) # kesuto nazeka segmentation fault okiru (b=8,16)
        sigma0 = random_tensor((b,b),["kl","kr"])
        ENV = bondenv.UnbridgeBondEnv(V, ["kl"], ["kr"])

        Ms,Ss,Ns,er1s,er2s = [],[],[],[],[]
        def wa(chi):
            memo = {}
            M,S,N = ENV.optimal_truncate(sigma0, chi=chi, memo=memo)
            self.assertTrue(memo["env_is_crazy_degenerated"])
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
            self.assertTrue(er2s[i] > er2s[i+1] or er2s[i+1] < 1e-8)

        self.assertAlmostEqual(er2s[c],0)
        self.assertAlmostEqual(er2s[b-1],0)

    # this is not omeate no kyodou. naosite~
    def test_truncatesisugi(self):
        b = 5
        n = 24
        jissai_chi = 4
        H = random_tensor((b,b,n),["kl","kr","extraction"])
        V = H * H.adjoint(["kl","kr"],style="aster")
        sigma0 = random_tensor((b,b),["kl","kr"])
        ENV = bondenv.UnbridgeBondEnv(V, ["kl"], ["kr"])

        Ms,Ss,Ns,er1s,er2s = [],[],[],[],[]
        def wa(chi):
            memo = {}
            M,S,N = ENV.optimal_truncate(sigma0, chi=chi, memo=memo)
            er1 = ((M*S*N)-sigma0).norm()
            er2 = (((M*S*N)-sigma0)*H).norm()
            Ms.append(M)
            Ss.append(S)
            Ns.append(N)
            er1s.append(er1)
            er2s.append(er2)

        for i in range(jissai_chi+2):
            wa(i+1)
        print(er2s)

        for i in range(jissai_chi+2):
            self.assertEqual(Ss[i].dim(0), min(i+1,jissai_chi))

        for i in range(jissai_chi+2):
            for j in range(min(i+1,jissai_chi)):
                self.assertGreater(Ss[i].data[j], 0)
                if j < min(i+1,jissai_chi)-1:
                    self.assertGreater(Ss[i].data[j], Ss[i].data[j+1])

        self.assertNotAlmostEqual(er2s[jissai_chi-2],0, places=6)
        self.assertNotAlmostEqual(er2s[jissai_chi-1],0, places=10) #hontoha motto ikeru noni!
        self.assertNotAlmostEqual(er2s[jissai_chi+1],0, places=10) #hontoha motto ikeru noni!


if __name__ == "__main__":
    unittest.main()
