import unittest
import sys
sys.path.append('../')
from tanuki import *
import numpy as np


class TestTensor(unittest.TestCase):
    def test_init(self):
        a = Tensor([[1,2],[3,4]],["a","b"])
        a = Tensor([[1,2],[3,4]],base_label="a")
        self.assertEqual(a.labels, ["a_0", "a_1"])
        a = Tensor([[1,2],[3,4]])
        with self.assertRaises(LabelsLengthError):
            a = Tensor([[1,2],[3,4]], ["a"])

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

    def test_labels(self):
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





if __name__=="__main__":
    unittest.main()