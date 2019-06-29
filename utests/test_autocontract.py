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
        ABC = actor.exec()
        self.assertEqual(ABC, A*(B*C))

        eternity = actor.get_eternity()
        print(eternity)
        self.assertEqual(eternity.cost, 2745)

    def test_input3(self):
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
        actor = AutoContractor(Ts)
        eternity = actor.get_eternity()
        print(eternity)

        self.assertEqual(eternity.shape, ())
        self.assertEqual(eternity.labels, [])
        self.assertEqual(eternity.cost, 115300100)

    def test_useful(self):
        A = random_tensor((5,5),["i","j"])
        B = random_tensor((5,7,8),["j","k","l"])
        C = random_tensor((7,8,9),["k","l","m"])
        actor = AutoContractor([A,B,C])
        ABC = actor.exec()
        self.assertEqual(ABC, A*(B*C))
        sitigosan = actor.exec([7,5,3])
        self.assertEqual(sitigosan, 105)
        D = random_tensor((9,5), ["m","i"])
        actor.exec([A,B,C,D], same_frame=False)
        self.assertEqual(actor.eternity.cost, 2770)
        
        actor = AutoContractor()
        for _ in range(3):
            A = random_tensor((5,5),["i","j"])
            B = random_tensor((5,7,8),["j","k","l"])
            C = random_tensor((7,8,9),["k","l","m"])
            ABC = actor.exec([A,B,C])
            self.assertEqual(ABC, A*B*C)
        print(actor)

    


if __name__=="__main__":
    unittest.main()