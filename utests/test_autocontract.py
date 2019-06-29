import unittest
import sys
sys.path.append('../')
from tanuki import *
from tanuki.autocontract import *
import numpy as np
from math import sqrt
import time



class TestAutoContract(unittest.TestCase):
    def test_input1(self):
        A = random_tensor((5,5),["i","j"])
        B = random_tensor((5,7,8),["j","k","l"])
        C = random_tensor((7,8,9),["k","l","m"])
        actor = AutoContractor([A,B,C])
        res = actor.get_eternity()
        print(res)
        
        self.assertEqual(res.cost, 2745)
        ABC = actor.exec()
        self.assertEqual(ABC, A*(B*C))



    """
    def test_netcon(self):
        A = lattices.random_fin1DSimBTPS([["p0"],["p1"]])#,["p2"],["p3"]])
        a = A.to_tensor()
        k = a * a.conjugate()
        B = A.adjoint()
        C = A.tensors+A.bdts+B.tensors+B.bdts
        #print(C)
        l = contract_all_common(C)
        self.assertEqual(k,l)
        #print(k,l)
    """
"""
class TestBrute(unittest.TestCase):
    def test_brute(self):
        A = TensorFrame((5,5),["i","j"], 0)
        B = TensorFrame((5,7,8),["j","k","l"], 1)
        C = TensorFrame((7,8,9),["k","l","m"], 2)
        hog = NetconBrute([A,B,C])
        res = hog.generate_eternity()
        
        self.assertEqual(res.cost, 2745)
        f = hog.generate_contractor()
        self.assertEqual(f(3,5,7), 105)

class TestJermyn(unittest.TestCase):
    def test_jermyn_input1(self):
        A = TensorFrame((5,5),["i","j"])
        B = TensorFrame((5,7,8),["j","k","l"])
        C = TensorFrame((7,8,9),["k","l","m"])
        hog = NetconJermyn([A,B,C])
        res = hog.generate_eternity()
        print(res)
        
        self.assertEqual(res.cost, 2745)
        f = hog.generate_contractor()
        self.assertEqual(f(3,5,7), 105)

    def test_jermyn_input3(self):
        rho = TensorFrame((10,10,10,10),["i","j","k","l"])
        o1 = TensorFrame((10,10,10),["i","f1","f2"])
        o2 = TensorFrame((10,10,10),["j","f3","f4"])
        o3 = TensorFrame((10,10,10),["k","f5","f6"])
        o4 = TensorFrame((10,10,10),["l","f7","f8"])
        u1 = TensorFrame((10,10,10,10),["f1","f6","g1","g2"])
        u2 = TensorFrame((10,10,10,10),["f5","f2","g3","g4"])
        u3 = TensorFrame((10,10,10,10),["f3","f8","g5","g6"])
        u4 = TensorFrame((10,10,10,10),["f7","f4","g3","g7"])
        op1 = TensorFrame((10,10),["g2","g6"])
        op2 = TensorFrame((10,10),["g1","g5"])
        op3 = TensorFrame((10,10),["g4","g7"])
        Ts = [rho,o1,o2,o3,o4,u1,u2,u3,u4,op1,op2,op3]
        hog = NetconJermyn(Ts)
        res = hog.generate_eternity()
        print(res)

        self.assertEqual(res.cost, 115300100)
"""

if __name__=="__main__":
    unittest.main()