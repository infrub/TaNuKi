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




def epm0300():
    registereds = []
    def register(algname, kwargs, color):
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
        registereds.append( (algname,kwargs,color,title) )

    register("LBOR", {}, "gray")
    register("NOR", {}, "brown")
    omegas = [1.6,1.65,1.7,1.75,1.8,1.85,1.9,1.95]
    for i,omega in enumerate(omegas):
        register("COR", {"omega":omega}, cm.gist_ncar((i+1)/(len(omegas)+1)))
    register("ROR", {"omega_cands":[1.6,1.65,1.7,1.72,1.74,1.76,1.78,1.8,1.82,1.84,1.94,1.95]}, "black")

    metaf = open("epm0300_oups/epm0300_meta.csv","a")
    metaf.write("b, chi, seed, ")
    for (_,_,_,title) in registereds:
        metaf.write(title+", ")
    metaf.write("\n")

    def ikuze(b,chi,seed):
        metaf.write(f"{b}, {chi}, {seed}, ")
        np.random.seed(seed=seed)
        random.seed(seed)

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
        M,S,N = ENV.optimal_truncate(A, maxiter=1000, conv_atol=1e-14, conv_rtol=1e-14, chi=chi, memo=memo, algname="LBOR")
        trueError = memo["sqdiff"]*(1-1e-10)

        for (algname, kwargs, color, title) in registereds:
            print(title, end=": ")
            try:
                memo = {}
                M,S,N = ENV.optimal_truncate(A, maxiter=400, chi=chi, memo=memo, algname=algname, **kwargs)
                df = pd.DataFrame({"y": memo.pop("sqdiff_history"), "x":np.arange(1,memo["iter_times"]+1)})
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
                print(f'{memo["iter_times"]}, {memo["sqdiff"]}')
                metaf.write(str(memo["iter_times"])+", ")
            except Exception as e:
                print(e)
                metaf.write("nan, ")

        metaf.write("\n")
        ax2.set_xlim(0,min(100,ax2.get_xlim()[1]))
        plt.legend()
        plt.suptitle(f"epm0300[b={b},chi={chi},seed={seed}]")
        plt.savefig(f"epm0300_oups/[b={b}, chi={chi}, seed={seed}].png", dpi=400)
        #plt.show()


    for chi in [1,3,4,6,7,9]:
        for seed in range(100):
            print(f"seed={seed}")
            ikuze(10,chi,seed)
            print("\n")



def epm0301():
    df = pd.read_csv("epm0300_oups/epm0300_meta.csv", index_col=False)
    print(df[df.chi==2].mean())
    print(df[df.chi==5].mean())
    print(df[df.chi==8].mean())


epm0300()