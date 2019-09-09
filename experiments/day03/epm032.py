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
from datetime import datetime

pd.options.display.max_columns = 30
pd.options.display.width = 160
MAX_PDF_PAGE = 100





def epm0320():
    registereds = []
    def register(algname, kwargs, color):
        if len(kwargs)==0:
            title = f"{algname}"
        else:
            title = f"{algname}("
            for kwarg in kwargs.values():
                if type(kwarg) in [float,np.float64]:
                    title += f"{kwarg:5.3f},"
                else:
                    title += f"{kwarg},"
            title = title[:-1] + ")"
        registereds.append( (algname,kwargs,color,title) )


    # 1 <= omegas < 2
    # o_c1s = 1/(2-omegas)
    # 1 <= o_c1s < inf
    o_c1s = np.linspace(1,20,20)
    omegas = 2 - 1/o_c1s

    register("ROR", {}, "black")
    for i,omega in enumerate(omegas):
        register("COR", {"omega":omega}, cm.gist_ncar((i+1)/(len(omegas)+1)))
    metaf = open("epm0310_oups/epm0310_meta.csv","a")
    metaf.write("b,chi,seed,trueError,")
    for (_,_,_,title) in registereds:
        metaf.write(title+",")
    metaf.write("\n")
    metaf.close()

    def ikuze(b,chi,seed):
        metaf = open("epm0320_oups/epm0320_meta.csv","a")

        print(f"b={b}, chi={chi}, seed={seed}, ", end="")
        metaf.write(f"{b},{chi},{seed},")
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
        M,S,N = ENV.optimal_truncate(A, maxiter=max(300,b*150), conv_atol=1e-14, conv_rtol=1e-14, chi=chi, memo=memo, algname="LBOR")
        trueError = memo["sqdiff"]
        print(f"trueError={trueError}")
        metaf.write(f"{trueError},")

        for (algname, kwargs, color, title) in registereds:
            print(title, end=": ")
            try:
                memo = {}
                M,S,N = ENV.optimal_truncate(A, maxiter=max(300,b*50), chi=chi, memo=memo, algname=algname, conv_atol=-1, conv_rtol=-1, conv_sqdiff=trueError*(1+1e-10), **kwargs)
                
                df = pd.DataFrame({"y": memo.pop("sqdiff_history"), "x":np.arange(1,memo["iter_times"]+2)})
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
                metaf.write(str(memo["iter_times"])+",")
            except Exception as e:
                print(e)
                metaf.write("nan,")

        metaf.write("\n")
        metaf.close()
        ax2.set_xlim(0,min(100,ax2.get_xlim()[1]))
        plt.legend()
        plt.suptitle(f"epm0320[b={b},chi={chi},seed={seed}]")
        #plt.savefig(f"epm0320_oups/[b={b}, chi={chi}, seed={seed}].png", dpi=400)
        plt.show()

    
    for b in [10]:
        if b <= 8:
            chis = np.linspace(1,b-1,b-1)
        else:
            chis = np.linspace(1,b-1,8)
        #for chi in chis:
        for chi in [8]: #omega=1.74 kurai ga best?
            chi = int(chi)
            for _ in range(1):
                seed = int(datetime.now().timestamp()*100) % 1000
                ikuze(b,chi,seed)
                print("\n")






epm0320()