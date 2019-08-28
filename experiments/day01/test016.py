import sys
sys.path.append('../../')
from tanuki import *
import numpy as np
import random
from colorama import Fore, Back, Style
import math



def test0160():
    for b in [10]: #range(1,31):
        for n in range(b,b**2+1):
            # juubun hyougen dekiru
            # <=> 4*b*chi    + chi    >= 2*n               + 2*chi*chi          + b
            #     (U,V free) (S free)    (mokuhyou jouken) (U,V unitary jouken) (henkaku muimi jiyuudo)
            avoiding_singular_chi = n // b
            exactly_solvable_chi = math.ceil((4*b+1 - math.sqrt((4*b+1)**2-8*b-16*n))/4)
            yoyuu = avoiding_singular_chi - exactly_solvable_chi
            print(f"{b:2},{n:3}: {avoiding_singular_chi:2} - {exactly_solvable_chi:2} = {yoyuu:2}")


def test0161():
    b = 20
    n = 500
    chi = 14
    H = random_tensor((b,b,n),["kl","kr","extraction"])
    V = H * H.adjoint(["kl","kr"],style="aster")
    sigma0 = random_tensor((b,b),["kl","kr"])
    ENV = bondenv.UnbridgeBondEnv(V, ["kl"], ["kr"])

    memo = {}
    M,S,N = ENV.optimal_truncate(sigma0, chi=chi, memo=memo)

    print(memo["elapsed_time"])


def test0162():
    for b in range(1,31):
        for n in range(1,b**2+1):
            for max_chi in range(1,b+3):
                avoiding_singular_chi = n // b
                exactly_solvable_chi = math.ceil((4*b+1 - math.sqrt((4*b+1)**2-8*b-16*n))/4)

                if exactly_solvable_chi <= max_chi:
                    chi = exactly_solvable_chi
                    exactly_solvable = True
                else:
                    chi = max_chi
                    exactly_solvable = False

                if n <= chi*b:
                    assert exactly_solvable
                else:
                    assert chi <= avoiding_singular_chi

#<==>

def test0163():
    for b in range(1,31):
        for n in range(1,b**2+1):
            for max_chi in range(1,b+3):
                avoiding_singular_chi = n // b
                exactly_solvable_chi = math.ceil((4*b+1 - math.sqrt((4*b+1)**2-8*b-16*n))/4)

                if exactly_solvable_chi <= max_chi:
                    chi = exactly_solvable_chi
                    exactly_solvable = True
                else:
                    chi = max_chi
                    exactly_solvable = False
                    assert n > chi*b
                if n>chi*b:
                    assert chi <= avoiding_singular_chi


#<==>

def test0164():
    for b in range(1,31):
        for n in range(1,b**2+1):
            for max_chi in range(1,b+3):
                avoiding_singular_chi = n // b
                exactly_solvable_chi = math.ceil((4*b+1 - math.sqrt((4*b+1)**2-8*b-16*n))/4)

                if exactly_solvable_chi > max_chi:
                    assert n/b > max_chi

                if exactly_solvable_chi <= max_chi:
                    chi = exactly_solvable_chi
                else:
                    chi = max_chi
                if n>chi*b:
                    assert chi <= avoiding_singular_chi


#<==>

def test0165():
    for b in range(1,31):
        for n in range(1,b**2+1):
            for max_chi in range(1,b+3):
                avoiding_singular_chi = n // b
                exactly_solvable_chi = math.ceil((4*b+1 - math.sqrt((4*b+1)**2-8*b-16*n))/4)

                if exactly_solvable_chi >= max_chi+1:
                    assert n/b > max_chi

                chi = min(exactly_solvable_chi, max_chi)
                if n>chi*b:
                    assert chi <= avoiding_singular_chi


#<==

def test0165():
    for b in range(1,31):
        for n in range(1,b**2+1):
            avoiding_singular_chi = n // b
            exactly_solvable_chi = math.ceil((4*b+1 - math.sqrt((4*b+1)**2-8*b-16*n))/4)

            assert n/b >= exactly_solvable_chi-1 #proven

            for max_chi in range(1,b+3):
                chi = min(exactly_solvable_chi, max_chi)
                if n>chi*b:
                    assert chi <= avoiding_singular_chi


#<==

def test0166():
    for b in range(1,31):
        for n in range(1,b**2+1):
            avoiding_singular_chi = n // b
            exactly_solvable_chi = math.ceil((4*b+1 - math.sqrt((4*b+1)**2-8*b-16*n))/4)

            assert n/b >= exactly_solvable_chi-1 #proven

            for chi in range(1,b+3):
                if n/b>chi:
                    assert chi <= avoiding_singular_chi


#<==

def test0167():
    for b in range(1,31):
        for n in range(1,b**2+1):
            avoiding_singular_chi = n // b
            exactly_solvable_chi = math.ceil((4*b+1 - math.sqrt((4*b+1)**2-8*b-16*n))/4)

            assert n/b >= exactly_solvable_chi-1 #proven

            for chi in range(1,b+3):
                if n//b>=chi:
                    assert chi <= avoiding_singular_chi #trivial

test0167()  