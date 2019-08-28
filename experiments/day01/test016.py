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


test0160()