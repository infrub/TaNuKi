import unittest
import sys
sys.path.append('../')
from tanuki import *
from tanuki.autocontract import *
import numpy as np
from math import sqrt
import time


class TestCyclicList(unittest.TestCase):
    def test01(self):
        a = CyclicList([2,3,4])
        self.assertEqual(a[0],2)
        self.assertEqual(a[1],3)
        self.assertEqual(a[3],2)
        self.assertEqual(a[301],3)
        self.assertEqual(a[-1],4)
        a[5] = 1
        self.assertEqual(a[2],1)
        b = CyclicList(a)
        b[0] = 0
        self.assertNotEqual(a[0],b[0]) 

class TestCollateralBool(unittest.TestCase):
    def test01(self):
        a = CollateralBool(True, "Alice")
        b = CollateralBool(False, "Bob")
        self.assertTrue(a)
        self.assertFalse(b)
        self.assertFalse(a&b)
        self.assertTrue(a|b)
        self.assertTrue(len(str(a)) > 6)









if __name__=="__main__":
    unittest.main()