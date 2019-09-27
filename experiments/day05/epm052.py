import sys,os
sys.path.append('../../')
from tanuki import *
import numpy as np
import scipy as sp
import scipy.optimize as spo
import random
from colorama import Fore, Back, Style
import math
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from datetime import datetime
import textwrap

pd.options.display.max_columns = 30
pd.options.display.width = 160
np.set_printoptions(linewidth=500)
tnc.display_max_size = float("inf")



class open_and_stdout:
    def __init__(self,filename):
        self.file = open(filename,"w")
        self.depth = 0
    def indent(self):
        self.depth += 1
    def dedent(self):
        self.depth -= 1
    def write(self,*args,sep=" ",end="\n"):
        text = sep.join(map(str,args)) + end
        text = textwrap.indent(text, "    "*self.depth)
        self.file.write(text)
        sys.stdout.write(text)
    def __enter__(self):
        return self
    def __exit__(self, ex_type, ex_value, trace):
        self.file.close()



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



def epm052_base(epmName, b, chi, epmlen):
    os.makedirs(f"{epmName}_oups/", exist_ok=True)
    metaf = open(f"{epmName}_oups/{epmName}_meta.csv","a")
    metaf.write("b,chi,seed,")
    for (_,_,_,title) in registereds:
        metaf.write(title+",")
    metaf.write("minError\n")
    metaf.close()

    def ikuze(b,chi,seed):
        metaf = open(f"{epmName}_oups/{epmName}_meta.csv","a")
        with open_and_stdout(f"{epmName}_oups/[b={b}, chi={chi}, seed={seed}].txt") as oupf:

            oupf.write(f"b={b}, chi={chi}, seed={seed}, ")
            metaf.write(f"{b},{chi},{seed},")
            np.random.seed(seed=seed)
            random.seed(seed)

            n = b*b
            H = random_tensor((b,b,n),["kl","kr","extraction"])
            V = H * H.adjoint(["kl","kr"],style="aster")
            A0 = random_tensor((b,b),["kl","kr"])

            fig, axs = plt.subplots(1, 2, sharey=True, figsize=(13,8))
            ax1,ax2 = tuple(axs)
            #ax1.set_xscale("log")
            ax1.set_ylabel("precision scale")
            ax1.set_xlabel("iteration times")
            ax2.set_xlabel("times spent for getting over the precision range (lefter is better)")

            minError = float("inf")
            maxError = float("-inf")
            dfd = {}
            for (algname, kwargs, color, title) in registereds:
                np.random.seed(seed=seed)
                random.seed(seed)
                oupf.write(title+":")
                oupf.indent()
                try:
                    ENV = bondenv.UnbridgeBondOptimalTruncator(V, ["kl"], ["kr"], A0)
                    ENV.set_params(max_iter=max(300,b*50), chi=chi, algname=algname, conv_atol=-1, conv_rtol=-1, **kwargs)
                    ENV.initialize()
                    M,S,N = ENV.run()
                    df = pd.DataFrame({"y": ENV.sqdiff_history, "x":np.arange(1,ENV.iter_times+2)})
                    dfd[title] = df
                    metaf.write(str(ENV.sqdiff)+",")
                    oupf.write(f"iter_times, elapsed_time, sqdiff: {ENV.iter_times}, {ENV.elapsed_time}, {ENV.sqdiff}")
                    oupf.write("MSN:",M*S*N)
                    oupf.write("grad_by_M:",ENV.grad_by_M)
                    oupf.write("grad:",ENV.grad)
                    hessian_M_M = ENV.hessian_by_M_M
                    #oupf.write("hessian_by_M_M:", hessian_M_M)
                    oupf.write("eigvals of hessian_by_M_M:", xp.linalg.eigvalsh(hessian_M_M))
                    hessian = ENV.hessian
                    #oupf.write("hessian:", hessian)
                    oupf.write("eigvals of hessian:", xp.linalg.eigvalsh(hessian))
                    if minError > ENV.sqdiff:
                        minError = ENV.sqdiff
                    if maxError < ENV.sqdiff:
                        maxError = ENV.sqdiff
                except:
                    raise
                """
                except ValueError as e:
                    oupf.write(e)
                    metaf.write("nan,")"""
                oupf.dedent()
            
            metaf.write("minError")

            for (algname, kwargs, color, title) in registereds:
                try:
                    df = dfd.get(title,None)
                    if df is None:
                        continue
                    df["my"] = df.y - minError + 1e-15
                    df["logx"] = np.log10(df.x)
                    df["logmy"] = np.log10(df.my)
                    df["dlogx"] = df.logx.shift(-1) - df.logx
                    df["dlogmy"] = df.logmy.shift(-1) - df.logmy
                    df["-dx__dlogmy"] = -1.0/df["dlogmy"] #== logmy wo herasunoni kakaru x. tiisai houga ii!
                    df["tadasing_cost2"] = df["-dx__dlogmy"]
                    df["smoothed_tadasing_cost2"] = df.tadasing_cost2.rolling(5, win_type="triang").mean().shift(-2)
                    ax1.plot(df.x, df.logmy, label=title, color=color)
                    ax2.plot(df.smoothed_tadasing_cost2, df.logmy, label=title, color=color) #migikara hidarini jikanha susumuyo
                except ValueError as e:
                    oupf.write(e)

            metaf.write("\n")
            metaf.close()
            ax2.set_xlim(0,min(100,ax2.get_xlim()[1]))
            #ax1.set_ylim(minError,maxError)
            plt.legend()
            plt.suptitle(f"{epmName}[b={b},chi={chi},seed={seed}]")
            plt.savefig(f"{epmName}_oups/[b={b}, chi={chi}, seed={seed}].png", dpi=400)
            #plt.show()


    for _ in range(epmlen):
        seed = int(datetime.now().timestamp()*100) % 1000
        ikuze(b,chi,seed)
        print("\n")



def epm0520():
    for algname in ["NOR"]:
        register(algname, {}, "black")
    i = 0
    for algname in ["WCOR","WROR","IWROR","WLBOR","WNAG","SNAG"]:
        register(algname, {}, cm.Paired(i/12))
        i += 2
    i = 1
    for algname in ["SSpiral"]:
        for spiral_turn_max in [8,11,14,17,20]:
            register(algname, {"spiral_turn_max":spiral_turn_max}, cm.gist_ncar(i/12))
            i += 2
    epm052_base("epm0520_test", 4, 2, 10)




epm0520()