import unittest
import sys
sys.path.append('../')
from tanuki import *
import numpy as np
from math import sqrt


class TestTensor(unittest.TestCase):
    def test_init(self):
        a = Tensor([[1,2],[3,4]],["a","b"])
        a = Tensor([[1,2],[3,4]],base_label="a")
        self.assertEqual(a.labels, ["a_0", "a_1"])
        a = Tensor([[1,2],[3,4]])
        with self.assertRaises(LabelsLengthError):
            a = Tensor([[1,2],[3,4]], ["a"])
        with self.assertRaises(LabelsTypeError):
            a = Tensor([[1,2],[3,4]], [8,9])

    def test_copy(self):
        d = np.array([[1,2],[3,4]])
        a1 = Tensor(d,["a","b"])
        a2 = a1.copy(shallow=False)
        a2.data[0,0]=6
        self.assertNotEqual(a1.data[0,0],a2.data[0,0])
        self.assertNotEqual(a2.data[0,0],d[0,0])
        a3 = a1.copy(shallow=True)
        a3.data[0,0]=7
        self.assertEqual(a1.data[0,0],a3.data[0,0])
        self.assertEqual(a3.data[0,0],d[0,0])

        d = np.array([[1,2],[3,4]])
        a1 = Tensor(d,["a","b"],copy=True)
        a2 = a1.copy(shallow=False)
        a2.data[0,0]=8
        self.assertNotEqual(a1.data[0,0],a2.data[0,0])
        self.assertNotEqual(a2.data[0,0],d[0,0])
        a3 = a1.copy(shallow=True)
        a3.data[0,0]=9
        self.assertEqual(a1.data[0,0],a3.data[0,0])
        self.assertNotEqual(a3.data[0,0],d[0,0])

    def test_instant(self):
        a = random_tensor((2,3,4))
        self.assertEqual(a.shape, (2,3,4))
        a = zeros_tensor((2,3,4))
        self.assertEqual(a.shape, (2,3,4))
        a = identity_tensor((2,3))
        self.assertEqual(a.shape, (2,3,2,3))
        a = identity_tensor((2,3),6)
        self.assertEqual(a.shape, (2,3,6))

    def test_indices_of_labels(self):
        a = random_tensor((2,3,4,5,6,7),["a","b","c","b","d","a"])
        self.assertEqual(a.labels_of_indices([2,3]),["c","b"])
        self.assertEqual(a.indices_of_labels_front(["b","c","b","a"]),[1,2,3,0])
        self.assertEqual(a.indices_of_labels_back(["b","c","d","a","b"]),[3,2,4,5,1])
        with self.assertRaises(ValueError):
            a.indices_of_labels(["b","b","b"])

    def test_normarg(self):
        a = random_tensor((2,3,4,5,6),["a","b",("c","b"),"c","a"])
        self.assertEqual(a.normarg_indices_front(["c","b"]),[3,1])
        self.assertEqual(a.normarg_indices_front([("c","b")]),[2])
        self.assertEqual(a.normarg_indices_front(("c","b")),[2])
        self.assertEqual(a.normarg_indices_back(4),[4])
        self.assertEqual(a.normarg_indices_front([0,"a"]),[0,4])
        self.assertEqual(a.normarg_indices_front([4,"a"]),[4,0])
        self.assertEqual(a.normarg_indices_back([0,"a"]),[0,4])
        self.assertEqual(a.normarg_indices_back([4,"a"]),[4,0])
        self.assertEqual(a.normarg_indices_front(["a","a"]),[0,4])
        self.assertEqual(a.normarg_indices_back(["a","a"]),[0,4])
        with self.assertRaises(ValueError):
            a.normarg_indices(["e"])
        with self.assertRaises(ValueError):
            a.normarg_indices(["a","a","a"])
        self.assertEqual(a.normarg_complement_indices([3,"a"]),([3,0],[1,2,4]))

    def test_replace_label(self):
        a = random_tensor((2,3,4,5,6),["a","b","c","d","e"])
        a.replace_labels("a","b")
        self.assertEqual(a.labels[0],"b")
        a = a.replace_labels(["b"],"c", inplace=False)
        self.assertEqual(a.labels[0],"c")
        a.replace_labels(1,"c")
        self.assertEqual(a.labels[1],"c")
        a.replace_labels(["c","c"],["a","a"])
        self.assertEqual(a.labels, ["a","a","c","d","e"])
        a.aster_labels(["a","c"])
        self.assertEqual(a.labels, ["a*","a","c*","d","e"])
        a.aster_labels(["a","a*"])
        self.assertEqual(a.labels, ["a**","a*","c*","d","e"])
        a.unaster_labels("a**")
        self.assertEqual(a.labels, ["a*","a*","c*","d","e"])
        a.unaster_labels("a*")
        self.assertEqual(a.labels, ["a","a*","c*","d","e"])

    def test_unary_ops(self):
        a = Tensor([[[1,2],[3,4]],[[0,3],[3,4]]])
        self.assertEqual(a.norm(), 8)
        a.normalize(inplace=True)
        d = xp.array([[[0.125,0.25],[0.375,0.5]],[[0.0,0.375],[0.375,0.5]]])
        self.assertTrue(xp.all(a.data == d))
        a = random_tensor((3,4,5),["a","b","c"],dtype=complex)
        self.assertFalse(a.is_hermite("a","b"))
        b = a.adjoint("a","b")
        c = b.adjoint("a","b")
        self.assertEqual(a,c)
        a = random_tensor((3,3,5),["a","b","c"],dtype=complex)
        a.hermite("a","b",inplace=True)
        self.assertTrue(a.is_hermite(["a"],["b"]))
        a.imag(inplace=True)
        self.assertEqual(a.data[0,0,0],0)
        self.assertEqual(a.data[1,1,0],0)
        self.assertEqual(a.data[1,1,1],0)

    def test_binary_ops(self):
        a1 = random_tensor((2,3,4),["a","b","c"])
        a2 = random_tensor_like(a1)
        b1 = random_tensor((4,3,4),["c","d","e"])
        self.assertEqual((a1+a2)*b1, a1*b1+a2*b1)
        b2 = random_tensor((4,3,5),["c","d","e"])
        self.assertNotEqual(a1, b1)
        self.assertNotEqual(b1, b2)
        self.assertFalse(a1.__eq__(a1.replace_labels("b","c",inplace=False), skipLabelSort=False))
        self.assertTrue(a1.__eq__(a1.replace_labels("b","c",inplace=False), skipLabelSort=True))

    def test_move_indices(self):
        a = random_tensor((2,3,4,5),["a","b","c","a"])
        b = a.move_all_indices(["b","a","a","c"])
        self.assertEqual(b.labels, ["b","a","a","c"])
        self.assertEqual(b.shape, (3,2,5,4))
        b.move_all_indices([1,0,3,2],inplace=True)
        self.assertEqual(a,b)

    def test_trace(self):
        a = Tensor([[[1,2],[3,4]],[[5,6],[7,8]]],["a","b","c"])
        b = a.trace(1,2)
        self.assertEqual(b, Tensor([5,13],["a"]))
        a = random_tensor((3,3,3,3,5),["a","a","a","a","c"])
        b = a.trace(0,2).trace(0,1)
        c = a.trace()
        d = a.trace([0,1],[2,3])
        self.assertEqual(b,c)
        self.assertEqual(b,d)

    def test_contract(self):
        a = random_tensor((3,4,5),["a","b","c"])
        b = random_tensor((5,4,3),["d","e","f"])
        c = random_tensor((3,4,5),["g","h","i"])
        self.assertEqual((a["c"]*b["d"])["f"]*c["g"], a["c"]*(b["f"]*c["g"])[["d"]])
        a2 = a.move_all_indices(["c","b","a"])
        self.assertEqual(a["b"]*b["e"], a2["b"]*b["e"])
        a = random_tensor((3,3,4,4),["a","a","b","b"])
        b = random_tensor((3,3,4,4),["c","c","b","b"])
        self.assertEqual(a*b, a[2,3]*b[2,3])
        self.assertNotEqual(a*b, a[2,3]*b[3,2])
        self.assertEqual((a*b).labels, ["a","a","c","c"])

    def test_truncate_pad(self):
        a = random_tensor((3,4,5,6),["a","b","c","c"])
        b = a.pad_indices(["c"],[(0,2)])
        c = b.truncate_index("c", 5)
        self.assertEqual(a,c)



class TestDiagonalTensor(unittest.TestCase):
    def test_instant(self):
        a = random_diagonalTensor((2,3,4))
        self.assertEqual(a.shape, (2,3,4,2,3,4))
        a = random_diagonalTensor((3,2),["a","b"])
        self.assertEqual(a.labels, ["a","b","a","b"])
        a = random_diagonalTensor((3,2),["a","b","a","d"])
        self.assertEqual(a.labels, ["a","b","a","d"])

    def test_move(self):
        a = random_diagonalTensor((2,3,4),["a","b","c","d","e","f"])
        b = a.move_all_indices(["e","c",0,"b",5,"d"])
        self.assertEqual(b.labels, ["e","c","a","b","f","d"])
        self.assertEqual(type(b), DiagonalTensor)
        with self.assertRaises(CantKeepDiagonalityError):
            c = a.move_all_indices_assuming_can_keep_diagonality(["e","c","f","b","a","d"])
        c = a.move_all_indices(["e","c","f","b","a","d"])
        self.assertEqual(c.labels, ["e","c","f","b","a","d"])
        self.assertEqual(c.shape, (3,4,4,3,2,2))
        self.assertEqual(type(c), Tensor)
        self.assertAlmostEqual(a.norm(), c.norm())
        d = a.move_half_all_indices_to_top(["e","c","a"])
        self.assertEqual(d,b)
        with self.assertRaises(InputLengthError):
            e = a.move_half_all_indices_to_top(["e","c"])
        with self.assertRaises(CantKeepDiagonalityError):
            e = a.move_half_all_indices_to_top(["e","c","b"])

    def test_sqrt_inv(self):
        a = random_diagonalTensor((2,3),["a","b"])
        a.abs(inplace=True)
        b = a.sqrt()
        self.assertEqual(b["a","b"]*b["a","b"], a)
        c = a.inv()
        d = identity_diagonalTensor((2,3),["a","b"])
        self.assertEqual(a["a","b"]*c["a","b"], d)

    def test_trace(self):
        a = random_diagonalTensor((2,3),["a","b"])
        b = a.trace("a","a")
        c = a.to_tensor()
        d = c.trace("a","a")
        self.assertEqual(type(b), DiagonalTensor)
        self.assertEqual(type(d), Tensor)
        self.assertEqual(b, d)
        a = random_diagonalTensor((2,3,3),["a","b","c","d","e","f"])
        b = a.trace("b","f",inplace=False)
        self.assertEqual(type(b),DiagonalTensor)
        c = b.trace("c","e",inplace=False)
        self.assertEqual(type(c),DiagonalTensor)
        d = a.to_tensor().trace(["b","e"],["f","c"])
        e = a.trace(["f","e"],["b","c"])
        self.assertEqual(type(d),Tensor)
        self.assertEqual(type(e),DiagonalTensor)
        self.assertTrue(d.is_diagonal("a"))
        self.assertEqual(c,d)
        self.assertEqual(c,e)



class TestContract(unittest.TestCase):
    def test_T_T(self):
        a = random_tensor((2,3,4,5,4,3),["a","b","c","d","e","f"])
        b = random_tensor((6,3,4,7,4,3),["h","b","c","g","e","f"])
        c = a["b","c"]*b["b","c"]
        d = (a["b"]*b["b"]).trace("c","c")
        self.assertEqual(c,d)

    def test_DT_DT(self):
        a = random_diagonalTensor((3,3),["b","c","q","r"])
        b = random_diagonalTensor((3,3),["b","c","y","z"])

        c = a["b","c"]*b["b","c"]
        self.assertEqual(type(c), DiagonalTensor)
        d = a.to_tensor()["b","c"]*b["b","c"]
        self.assertEqual(type(d), Tensor)
        e = a.to_tensor()["b","c"]*b.to_tensor()["b","c"]
        self.assertEqual(type(e), Tensor)
        f = a["b","c"]*b.to_tensor()["b","c"]
        g = a*b
        self.assertEqual(type(f), Tensor)
        self.assertEqual(c,d)
        self.assertEqual(c,e)
        self.assertEqual(e,f)
        self.assertEqual(f,g)

        c = a["b","c"]*b["c","b"]
        self.assertEqual(type(c), DiagonalTensor)
        d = a.to_tensor()["b","c"]*b["c","b"]
        self.assertEqual(type(d), Tensor)
        e = a.to_tensor()["b","c"]*b.to_tensor()["c","b"]
        self.assertEqual(type(e), Tensor)
        f = a["b","c"]*b.to_tensor()["c","b"]
        self.assertEqual(type(f), Tensor)
        self.assertEqual(c,d)
        self.assertEqual(c,e)
        self.assertEqual(e,f)

        c = a["b"]*b["c"]
        self.assertEqual(type(c), DiagonalTensor)
        d = a.to_tensor()["b"]*b.to_tensor()["c"]
        self.assertEqual(c,d)



if __name__=="__main__":
    unittest.main()