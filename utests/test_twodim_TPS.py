import unittest
import sys
sys.path.append('../')
from tanuki import *
from tanuki.onedim import *
from tanuki.twodim import *
import numpy as np
from math import sqrt
import copy



def make_random_TPS():
    b = 2
    chi = 20
    A = random_tensor((b,chi,chi,chi,chi),["a","al","ar","au","ad"])
    B = random_tensor((b,chi,chi,chi,chi),["b","bl","br","bu","bd"])
    L = random_diagonalTensor((chi,),["al","bl"])
    R = random_diagonalTensor((chi,),["ar","br"])
    U = random_diagonalTensor((chi,),["au","bu"])
    D = random_diagonalTensor((chi,),["ad","bd"])

    return twodim.Ptn2DCheckerBTPS(A,B,L,R,U,D, width_scale=4, height_scale=4)




class TestOpn1DBTPS(unittest.TestCase):
    def test_super_orthogonalize_normalize(self):
        Z = make_random_TPS()
        z0 = Z.A*Z.L*Z.R*Z.U*Z.D*Z.B
        weight = Z.super_orthogonalize(normalize=True)
        z1 = Z.A*Z.L*Z.R*Z.U*Z.D*Z.B
        self.assertAlmostEqual(z0, z1*weight)

        conv_rtol = 1e-8
        conv_atol = 1e-11
        A,B,L,R,U,D = Z.A,Z.B,Z.L,Z.R,Z.U,Z.D
        for temp in [(A*R*U*D).is_prop_right_semi_unitary(rows=L, check_rtol=conv_rtol*1e3, check_atol=conv_atol*1e3),
                    (A*L*U*D).is_prop_right_semi_unitary(rows=R, check_rtol=conv_rtol*1e3, check_atol=conv_atol*1e3),
                    (A*L*R*D).is_prop_right_semi_unitary(rows=U, check_rtol=conv_rtol*1e3, check_atol=conv_atol*1e3),
                    (A*L*R*U).is_prop_right_semi_unitary(rows=D, check_rtol=conv_rtol*1e3, check_atol=conv_atol*1e3),
                    (B*R*U*D).is_prop_right_semi_unitary(rows=L, check_rtol=conv_rtol*1e3, check_atol=conv_atol*1e3),
                    (B*L*U*D).is_prop_right_semi_unitary(rows=R, check_rtol=conv_rtol*1e3, check_atol=conv_atol*1e3),
                    (B*L*R*D).is_prop_right_semi_unitary(rows=U, check_rtol=conv_rtol*1e3, check_atol=conv_atol*1e3),
                    (B*L*R*U).is_prop_right_semi_unitary(rows=D, check_rtol=conv_rtol*1e3, check_atol=conv_atol*1e3)]:
            self.assertTrue(temp)
            self.assertAlmostEqual(temp["factor"],1.0)
        for temp in [L,R,U,D]:
            self.assertAlmostEqual(L.norm(), 1.0)

    def test_super_orthogonalize_dont_normalize(self):
        Z = make_random_TPS()
        z0 = Z.A*Z.L*Z.R*Z.U*Z.D*Z.B
        weight = Z.super_orthogonalize(normalize=False)
        z1 = Z.A*Z.L*Z.R*Z.U*Z.D*Z.B
        self.assertAlmostEqual(z0, z1)
        self.assertEqual(weight, 1.0)

        conv_rtol = 1e-8
        conv_atol = 1e-11
        A,B,L,R,U,D = Z.A,Z.B,Z.L,Z.R,Z.U,Z.D
        for temp in [(A*R*U*D).is_prop_right_semi_unitary(rows=L, check_rtol=conv_rtol*1e3, check_atol=conv_atol*1e3),
                    (A*L*U*D).is_prop_right_semi_unitary(rows=R, check_rtol=conv_rtol*1e3, check_atol=conv_atol*1e3),
                    (A*L*R*D).is_prop_right_semi_unitary(rows=U, check_rtol=conv_rtol*1e3, check_atol=conv_atol*1e3),
                    (A*L*R*U).is_prop_right_semi_unitary(rows=D, check_rtol=conv_rtol*1e3, check_atol=conv_atol*1e3),
                    (B*R*U*D).is_prop_right_semi_unitary(rows=L, check_rtol=conv_rtol*1e3, check_atol=conv_atol*1e3),
                    (B*L*U*D).is_prop_right_semi_unitary(rows=R, check_rtol=conv_rtol*1e3, check_atol=conv_atol*1e3),
                    (B*L*R*D).is_prop_right_semi_unitary(rows=U, check_rtol=conv_rtol*1e3, check_atol=conv_atol*1e3),
                    (B*L*R*U).is_prop_right_semi_unitary(rows=D, check_rtol=conv_rtol*1e3, check_atol=conv_atol*1e3)]:
            self.assertTrue(temp)
            self.assertAlmostEqual(temp["factor"],1.0)



if __name__=="__main__":
    unittest.main()
