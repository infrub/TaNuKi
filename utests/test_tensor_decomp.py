import unittest
import sys
sys.path.append('../')
from tanuki import *
import numpy as np
from math import sqrt


class TestDecomp(unittest.TestCase):
    def test_normarg(self):
        a = normarg_svd_labels(None)
        self.assertEqual(a[0],a[1])
        self.assertEqual(a[1],a[2])
        self.assertEqual(a[2],a[3])
        a = normarg_svd_labels("funi")
        self.assertEqual(a,["funi","funi","funi","funi"])
        a = normarg_svd_labels(["funi","neko"])
        self.assertEqual(a,["funi","funi","neko","neko"])
        a = normarg_svd_labels(["funi","neko","neko","nyanko"])
        self.assertEqual(a,["funi","neko","neko","nyanko"])
        with self.assertRaises(ValueError):
            a = normarg_svd_labels(["funi","neko","nyanko"])

    def test_svd(self):
        A = random_tensor((2,3,4,3,2),["a","b","c","d","e"],dtype=complex)
        U,S,V = tensor_svd(A,["a","b"],svd_labels=["h","i"])
        self.assertTrue(U.is_semi_unitary(["a","b"]))
        self.assertTrue(S.is_diagonal("h"))
        self.assertTrue(V.is_semi_unitary(["c","d","e"]))
        self.assertFalse(V.is_semi_unitary(["c","e"]))
        self.assertEqual(A, U*S*V)
        self.assertNotEqual(A, U*V)
        self.assertEqual(A*U.conj()*V.conj(), S)
        self.assertEqual(type(S), DiagonalTensor)
        self.assertEqual(S,S.real())
        self.assertEqual(S.shape, (6,6))
        V2,S2,U2 = tensor_svd(A,["c","d","e"],svd_labels=["i","h"])
        self.assertEqual(S,S2)

    def test_tensor_svd(self):
        A = random_tensor((10,10,10,10),["a","b","c","d"])
        U,S,V = tensor_svd(A,["a","b"],svd_labels="h",chi=99)
        self.assertTrue(U.is_semi_unitary(["a","b"]))
        self.assertTrue(S.is_diagonal("h"))
        self.assertTrue(V.is_semi_unitary(["c","d"]))
        self.assertEqual(U.shape, (10,10,99))
        self.assertEqual(S.shape, (99,99))
        self.assertEqual(V.shape, (99,10,10))
        taikaNorm = (A-U*S*V).norm()
        U,S,V = tensor_svd(A,["a","b"],svd_labels="h",chi=100)
        notTaikaNorm = (A-U*S*V).norm()
        self.assertLess(taikaNorm, A.norm()/100)
        self.assertLess(notTaikaNorm, A.norm()/1000000)
        self.assertGreater(taikaNorm, notTaikaNorm)
        self.assertGreaterEqual(S.data[0],S.data[1])
        self.assertGreaterEqual(S.data[1],S.data[2])
        self.assertGreaterEqual(S.data[97],S.data[98])
        self.assertGreaterEqual(S.data[98],S.data[99])
        self.assertGreaterEqual(S.data[99],0)
        U,S,V = tensor_svd(A,["a","b"],svd_labels="h",atol=1)
        self.assertLess(S.halfsize, 100)
        self.assertGreater(S.halfsize, 80)

    def test_qr(self):
        A = random_tensor((7,8,9,10),["a","b","c","d"])
        Q, R = tensor_qr(A, ["a","b"], qr_labels="uni")
        self.assertEqual(A, Q*R)
        self.assertTrue(Q.is_unitary(["a","b"]))
        self.assertTrue(Q.is_unitary("uni"))
        self.assertTrue(R.is_triu("uni",["c","d"]))
        self.assertFalse(R.is_tril("uni", ["c","d"]))
        self.assertAlmostEqual(A.norm(), R.norm())
        A = random_tensor((10,9,8,7),["a","b","c","d"])
        L, Q = tensor_lq(A, ["a","b"], lq_labels="uni")
        self.assertEqual(A, L*Q)
        self.assertTrue(Q.is_unitary("uni"))
        self.assertTrue(Q.is_unitary(["c","d"]))
        self.assertTrue(L.is_tril(["a","b"],"uni"))
        self.assertFalse(L.is_triu(["a","b"],"uni"))
        self.assertAlmostEqual(A.norm(), L.norm())

    def test_eigh(self):
        A = random_tensor((7,8,8,7),["a","b","c","d"])
        A = A.hermite(["a","b"],["d","c"])
        self.assertTrue(A.is_hermite(["a","b"], ["d","c"]))
        V, W, Vh = tensor_eigh(A, ["a","b"], ["d","c"], eigh_labels="e")
        self.assertEqual(V*W*Vh, A)
        Vh.replace_labels(["c","d"],["b","a"])
        self.assertEqual(Vh, V.conj())
        self.assertTrue(V.is_unitary(["a","b"]))
        self.assertAlmostEqual(W.norm(), A.norm())

    def test_eigsh(self):
        A = Tensor([[4.0,2],[2,1]], ["a","b"])
        w, V = tensor_eigsh(A, ["a"])
        self.assertEqual(w, 5)
        self.assertAlmostEqual(abs(V.data[0]), 2/sqrt(5))
        self.assertAlmostEqual(abs(V.data[1]), 1/sqrt(5))
        V = random_tensor((100,),["a"])
        V.normalize(inplace=True)
        A = direct_product(V, V)
        A.data[0,0]=0
        w, V = tensor_eigsh(A, ["a"])
        self.assertNotEqual(1,w)
        self.assertAlmostEqual(1,w,3)



if __name__=="__main__":
    unittest.main()