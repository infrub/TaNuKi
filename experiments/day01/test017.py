import sys
sys.path.append('../../')
from tanuki import *
import numpy as np
import random
from colorama import Fore, Back, Style
import math



def test0170():
    for b in [10]: #range(1,31):
        for n in range(1,b**2+1):
            # juubun hyougen dekiru
            # <=> 4*b*chi    -2*chi*chi    >= 2*n             
            #     (M,N free) (M,N muimi)    (mokuhyou jouken) 
            avoiding_singular_chi = n // b
            exactly_solvable_chi = math.ceil(b - math.sqrt(b**2-n))
            yoyuu = avoiding_singular_chi - exactly_solvable_chi
            #if yoyuu < 0:
            #if avoiding_singular_chi == 1:
            if True:
                print(f"{b:2},{n:3}: {avoiding_singular_chi:2} - {exactly_solvable_chi:2} = {yoyuu:2}")




test0170()