import sys
sys.path.append('../../')
from tanuki import *
import numpy as np

def test0110():
    def f(b,n):
        H = random_tensor((b,b,n),["kl","kr","extraction"])
        V = H * H.adjoint(["kl","kr"],style="aster")
        sigma0 = random_tensor((b,b),["kl","kr"])
        ENV = bondenv.UnbridgeBondEnv(V, ["kl"], ["kr"])

        memo = {}
        M,S,N = ENV.optimal_truncate(sigma0, chi=b, memo=memo)

        if memo["is_crazy_singular"]:
            return 0
        return memo["chi"]

    def yosou(b,n):
        n = min(n,b**2)
        return n//b


    bs = list(range(1,7))
    ns = list(range(1,38))
    chiss = [[0 for n in [0]+ns] for b in [0]+bs]
    for b in bs:
        for n in ns:
            chiss[b][n] = f(b,n)

    print("b\\n", end="")
    for n in ns:
        print(f"{n:2}", end=" ")
    print()
    for b in bs:
        print(f"{b:2}", end=" ")
        for n in ns:
            print(f"{chiss[b][n]:2}", end=" ")
        print()

    print()

    print("b\\n", end="")
    for n in ns:
        print(f"{n:2}", end=" ")
    print()
    for b in bs:
        print(f"{b:2}", end=" ")
        for n in ns:
            print(f"{yosou(b,n):2}", end=" ")
        print()

    print()

    print("b\\n", end="")
    for n in ns:
        print(f"{n:2}", end=" ")
    print()
    for b in bs:
        print(f"{b:2}", end=" ")
        for n in ns:
            print(f"{yosou(b,n)-chiss[b][n]:2}", end=" ")
        print()


def test0111():
    for b in range(1,31):
        for n in range(b,b**2+1):
            juubun_chi = (4*b+1 - np.sqrt((4*b+1)**2-16*n))/4 # jouken to jiyuudo kara koredake areba itti suru to omou
            dekakute_chi = n / b
            jissai_chi = max(1, n // b)
            assert dekakute_chi >= juubun_chi # this is proven
            assert jissai_chi >= juubun_chi
            if jissai_chi -1 >= juubun_chi:
                print(b,n,jissai_chi - juubun_chi, "mada ikeru")
        print()


def test0112():
    def f(b,n,chi):
        H = random_tensor((b,b,n),["kl","kr","extraction"])
        V = H * H.adjoint(["kl","kr"],style="aster")
        sigma0 = random_tensor((b,b),["kl","kr"])
        ENV = bondenv.UnbridgeBondEnv(V, ["kl"], ["kr"])

        memo = {}
        M,S,N = ENV.optimal_truncate(sigma0, chi=chi, memo=memo)
        nrm = ((M*S*N-sigma0)*H).norm()

        print(b,n,chi,"->",memo["chi"],memo["iter_times"], nrm, ("small" if nrm < 1e-6 else "LARGE"))

    print()
    print("5 20 1.5 mada ikeru")
    f(5,20,5)
    f(5,20,4)
    f(5,20,3)
    f(5,20,2)
    print()
    print("8 29 1.0 mada ikeru")
    f(8,29,4)
    f(8,29,3)
    f(8,29,2)
    f(8,29,1)
    print()
    print("9 63 2.5 mada ikeru")
    f(9,63,8)
    f(9,63,7)
    f(9,63,6)
    f(9,63,5)
    f(9,63,4)
    print()
    print("9 74 2.1503676271838614 mada ikeru")
    f(9,74,9)
    f(9,74,8)
    f(9,74,7)
    f(9,74,6)
    f(9,74,5)
    print()
    print("30 726 7.5 mada ikeru")
    f(30,726,25)
    f(30,726,24)
    f(30,726,23)
    f(30,726,22)
    f(30,726,21)
    f(30,726,20)
    f(30,726,19)
    f(30,726,18)
    f(30,726,17)
    f(30,726,16)

""" test0122()
5 20 1.5 mada ikeru
5 20 5 -> 4 2 4.408038209596914e-14 small
5 20 4 -> 4 1 2.1159766437369385e-14 small
5 20 3 -> 3 176 4.985740823417015e-09 small
5 20 2 -> 2 86 1.823695366637146 LARGE

8 29 1.0 mada ikeru
8 29 4 -> 3 56 5.1038276611795604e-09 small
8 29 3 -> 3 28 2.1490575355253804e-09 small
8 29 2 -> 2 566 0.19363618023158166 LARGE #e-
8 29 1 -> 1 125 12.493163101697505 LARGE

9 63 2.5 mada ikeru
9 63 8 -> 7 2 8.78010580116332e-12 small
9 63 7 -> 7 1 3.957726993162618e-13 small
9 63 6 -> 6 76 5.8824997135583495e-09 small
9 63 5 -> 5 355 4.409310744331602e-08 small
9 63 4 -> 4 765 2.838677619951649 LARGE

9 74 2.1503676271838614 mada ikeru
9 74 9 -> 8 43 1.1137892873280094e-09 small
9 74 8 -> 8 15 1.6836991791348055e-10 small
9 74 7 -> 7 265 8.969751483751945e-09 small
9 74 6 -> 6 999 0.40430361418315147 LARGE #e-
9 74 5 -> 5 463 2.8329513906928594 LARGE

30 726 7.5 mada ikeru
30 726 25 -> 24 10 3.276269010959925e-10 small
30 726 24 -> 24 11 1.8623967303108163e-10 small
30 726 23 -> 23 27 7.3406893521801085e-09 small
30 726 22 -> 22 65 3.3138302318449745e-08 small
30 726 21 -> 21 144 7.680073737922638e-08 small
30 726 20 -> 20 264 1.4532666014727833e-07 small
30 726 19 -> 19 602 2.459265861013422e-07 small
30 726 18 -> 18 999 0.00045366364305355077 LARGE #e-
30 726 17 -> 17 999 0.822190459277258 LARGE #e-
30 726 16 -> 16 999 5.106705528015934 LARGE
"""

test0111()