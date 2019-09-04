import sys
sys.path.append('../../')
from tanuki import *
import numpy as np
import scipy as sp
import scipy.optimize as spo
import random
from colorama import Fore, Back, Style
import math
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import pandas as pd

pd.options.display.max_columns = 20
pd.options.display.width = 160
MAX_PDF_PAGE = 100







# rantaku tameso-
def epm0240():
    def ikuze(seed):
        np.random.seed(seed=seed)

        b,chi = 10,8
        n = b*b
        H = random_tensor((b,b,n),["kl","kr","extraction"])
        V = H * H.adjoint(["kl","kr"],style="aster")
        A = random_tensor((b,b),["kl","kr"])
        ENV = bondenv.UnbridgeBondEnv(V, ["kl"], ["kr"])

        fig, axs = plt.subplots(1, 2, sharey=True, figsize=(13,8))
        ax1,ax2 = tuple(axs)
        ax1.set_xscale("log")
        ax1.set_ylabel("precision scale")
        ax1.set_xlabel("iteration times")
        ax2.set_xlabel("times spent for getting over the precision range (lefter is better)")

        memo = {}
        M,S,N = ENV.optimal_truncate(A, maxiter=1000, conv_atol=1e-14, conv_rtol=1e-14, chi=chi, memo=memo, algname="alg04")
        trueError = memo["sq_diff"]*(1-1e-10)

        def yaru(algname, kwargs, color):
            if len(kwargs)==0:
                title = f"{algname}"
            else:
                title = f"{algname}("
                for kwarg in kwargs.values():
                    if type(kwarg) == float:
                        title += f"{kwarg:4.2f},"
                    else:
                        title += f"{kwarg},"
                title = title[:-1] + ")"
            print(title)
            memo = {}
            M,S,N = ENV.optimal_truncate(A, maxiter=400, chi=chi, memo=memo, algname=algname, **kwargs)
            df = pd.DataFrame({"y": memo.pop("fxs"), "x":np.arange(1,memo["iter_times"]+1)})
            df["my"] = df.y - trueError
            df["logx"] = np.log(df.x)
            df["logmy"] = np.log(df.my)
            df["dlogx"] = df.logx.shift(-1) - df.logx
            df["dlogmy"] = df.logmy.shift(-1) - df.logmy
            df["-dx__dlogmy"] = -1.0/df["dlogmy"] #== logmy wo herasunoni kakaru x. tiisai houga ii!
            df["tadasing_cost2"] = df["-dx__dlogmy"]
            df["smoothed_tadasing_cost2"] = df.tadasing_cost2.rolling(5, win_type="triang").mean().shift(-2)


            ax1.plot(df.x, df.logmy, label=title, color=color)
            ax2.plot(df.smoothed_tadasing_cost2, df.logmy, label=title, color=color) #migikara hidarini jikanha susumuyo
            #print(S)
            print(memo)
            #print()

        yaru("alg04", {}, "gray")

        kasokus = [1.6,1.65,1.7,1.75,1.8,1.85,1.9,1.94]
        for i,kasoku in enumerate(kasokus):
            try:
                yaru("alg08", {"kasoku":kasoku}, cm.gist_ncar((i+1)/(len(kasokus)+1)))
            except Exception as e:
                print(e)

        try:
            yaru("alg21", {"kasokus":kasokus}, "black")
        except Exception as e:
            print(e)

        
        ax2.set_xlim(0,min(100,ax2.get_xlim()[1]))
        plt.legend()
        plt.suptitle(f"epm0240[b={b},chi={chi},seed={seed}]")
        plt.savefig(f"epm0240_oups/[b={b}, chi={chi}, seed={seed}].png", dpi=400)
        #plt.show()

    for seed in range(100):
        print(f"seed={seed}")
        ikuze(seed)
        print("\n")


epm0240()