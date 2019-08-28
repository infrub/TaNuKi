import sys
sys.path.append('../../')
from tanuki import *
import numpy as np
import random
from colorama import Fore, Back, Style



def test0150():
    for b in range(1,31):
        for n in range(b,b**2+1):
            # juubun hyougen dekiru
            # <=> 4*b*chi    + chi    >= 2*n               + 2*chi*chi          + b
            #     (U,V free) (S free)    (mokuhyou jouken) (U,V unitary jouken) (henkaku muimi jiyuudo)
            juubun_chi = (4*b+1 - np.sqrt((4*b+1)**2-8*b-16*n))/4
            dekakute_chi = n / b
            jissai_chi = max(1, n // b)
            yoyuu = jissai_chi - juubun_chi
            if yoyuu < 0:
                print(f"{b:2},{n:3}: {jissai_chi:2} - {juubun_chi:6.3f} = {yoyuu:2.3f}")
        print()



def test0151():
    for b in range(1,31):
        for n in range(b,b**2+1):
            B = 4*b
            N = 4*n
            P = N // B
            Q = N % B
            p = n // b
            q = n % b
            ## I'm sure mondai >= 0
            #mondai = (4*b+1 - np.sqrt((4*b+1)**2-8*b-16*n))/4 - p - 1/4
            #mondai = np.sqrt(B*B+1-4*P*B-4*Q) - (B-4*P)
            #mondai = (4*P*B+1) - (16*P*P+4*Q)
            mondai = (p*b) - (p+q)
            if mondai < 1:
                print(p,q,mondai)


def f(b,n):
    H = random_tensor((b,b,n),["kl","kr","extraction"])
    V = H * H.adjoint(["kl","kr"],style="aster")
    sigma0 = random_tensor((b,b),["kl","kr"])
    ENV = bondenv.UnbridgeBondEnv(V, ["kl"], ["kr"])

    memo = {}
    M,S,N = ENV.optimal_truncate(sigma0, chi=b, memo=memo)

    return memo["chi"], (((M*S*N)-sigma0)*H).norm()



def test0152():
    bns = [(7,13),(7,14),(13,24),(13,25),(13,26),(26,39),(26,51)]
    for (b,n) in bns:
        if n % 4 == 0:
            continue
        juubun_chi = (4*b+1 - np.sqrt((4*b+1)**2-8*b-16*n))/4
        jissai_chi = max(1, n // b)
        yoyuu = jissai_chi - juubun_chi
        chi,gosa = f(b,n)

        if yoyuu>=0:
            if gosa<=1e-8:
                print(Fore.WHITE,end="")
            else:
                print(Fore.RED,end="") #nande gosa dekai nen (majide yabai)
        else:
            if gosa<=1e-8:
                print(Fore.BLUE,end="") #nande gosa tiisai nen (iya iikedoyo)
            else:
                print(Fore.GREEN,end="") #sou naruyone~~

        print(b,n,chi,yoyuu,gosa)
        print(Fore.RESET,end="")




def test0153():
    for b in range(1,31):
        for n in random.sample(range(b,b*b+1), b):
            if n % 4 == 0:
                continue
            juubun_chi = (4*b+1 - np.sqrt((4*b+1)**2-8*b-16*n))/4
            jissai_chi = max(1, n // b)
            yoyuu = jissai_chi - juubun_chi
            chi,gosa = f(b,n)

            if yoyuu>=0:
                if gosa<=1e-8:
                    print(Fore.WHITE,end="")
                else:
                    print(Fore.RED,end="") #nande gosa dekai nen (majide yabai)
            else:
                if gosa<=1e-8:
                    print(Fore.BLUE,end="") #nande gosa tiisai nen (iya iikedoyo)
                else:
                    print(Fore.GREEN,end="") #sou naruyone~~

            print(b,n,chi,yoyuu,gosa)
            print(Fore.RESET,end="")


test0150()