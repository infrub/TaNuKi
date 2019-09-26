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

pd.options.display.max_columns = 30
pd.options.display.width = 160
MAX_PDF_PAGE = 100



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



def epm041_base(epmName, b, chi, epmlen):
    os.makedirs(f"{epmName}_oups/", exist_ok=True)
    metaf = open(f"{epmName}_oups/{epmName}_meta.csv","a")
    metaf.write("b,chi,seed,")
    for (_,_,_,title) in registereds:
        metaf.write(title+",")
    metaf.write("minError\n")
    metaf.close()

    def ikuze(b,chi,seed):
        metaf = open(f"{epmName}_oups/{epmName}_meta.csv","a")

        print(f"b={b}, chi={chi}, seed={seed}, ")
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
            print(title, end=": ")
            try:
                memo = {}
                M,S,N = ENV.optimal_truncate(A, maxiter=max(300,b*50), chi=chi, memo=memo, algname=algname, conv_atol=-1, conv_rtol=-1, **kwargs)
                
                df = pd.DataFrame({"y": memo.pop("sqdiff_history"), "x":np.arange(1,memo["iter_times"]+2)})
                dfd[title] = df
                print(f'{memo["iter_times"]}, {memo["elapsed_time"]}, {memo["sqdiff"]}')
                metaf.write(str(memo["sqdiff"])+",")
                if minError > memo["sqdiff"]:
                    minError = memo["sqdiff"]
                if maxError < memo["sqdiff"]:
                    maxError = memo["sqdiff"]
            except Exception as e:
                print(e)
                metaf.write("nan,")
        
        metaf.write("minError")

        for (algname, kwargs, color, title) in registereds:
            try:
                df = dfd[title]
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
            except Exception as e:
                print(e)

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



def epm0410():
    i = 0
    for algname in ["NOR","COR","ROR","IROR","LBOR"]:
        register(algname, {}, cm.Paired(i/12))
        i += 2
    epm041_base("epm0410", 12, 6, 10)

def epm0411():
    i = 0
    for algname in ["NOR","COR","ROR","IROR","LBOR"]:
        register(algname, {}, cm.Paired(i/12))
        i += 2
    i = 1
    for algname in ["MSGDoid"]:
        register(algname, {}, cm.Paired(i/12))
        i += 2
    epm041_base("epm0411", 12, 6, 10)

def epm0412():
    i = 0
    for algname in ["NOR","COR","ROR","IROR","LBOR"]:
        register(algname, {}, cm.Paired(i/12))
        i += 2
    i = 1
    for algname in ["MSGDoid"]:
        for emamu in [0.1,0.3,0.5,0.7,0.9]:
            register(algname, {"emamu":emamu}, cm.Paired(i/12))
            i += 2
    epm041_base("epm0412", 12, 6, 10)

def epm0413():
    i = 0
    for algname in ["NOR","COR","ROR","IROR","LBOR"]:
        register(algname, {}, cm.Paired(i/12))
        i += 2
    i = 1
    for algname in ["MSGDoid"]:
        for emamu in [0.86,0.88,0.90,0.92,0.94]:
            register(algname, {"emamu":emamu}, cm.Paired(i/12))
            i += 2
    epm041_base("epm0413_kari", 12, 6, 10)

def epm0414():
    i = 0
    for algname in ["NOR","COR","ROR","IROR","LBOR","MSGDoid"]:
        register(algname, {}, cm.Paired(i/12))
        i += 2
    i = 1
    for algname in ["NAGoid"]:
        for emamu in [0.86,0.88,0.90,0.92,0.94]:
            register(algname, {"emamu":emamu}, cm.Paired(i/12))
            i += 2
    epm041_base("epm0414", 12, 6, 100)

def epm0415():
    for algname in ["NOR"]:
        register(algname, {}, "black")

    i = 0
    for algname in ["WCOR","WROR","IWROR","WLBOR","WMSGD","WNAG"]:
        register(algname, {}, cm.Paired(i/12))
        i += 2
    i = 1
    for algname in ["SCOR","SROR","","","SMSGD","SNAG"]:
        if algname != "":
            register(algname, {}, cm.Paired(i/12))
        i += 2
    epm041_base("epm0415", 12, 6, 1)

def epm0416():
    for algname in ["NOR"]:
        register(algname, {}, "black")

    i = 0
    for algname in ["WCOR","WROR","IWROR","WLBOR","WMSGD","WNAG"]:
        register(algname, {}, cm.Paired(i/12))
        i += 2
    i = 1
    for algname in ["SCOR","SROR","SSpiral","","SMSGD","SNAG"]:
        if algname != "":
            register(algname, {}, cm.Paired(i/12))
        i += 2
    epm041_base("epm0416", 12, 6, 10)

def epm0417():
    for algname in ["NOR"]:
        register(algname, {}, "black")

    i = 0
    for algname in ["WCOR","WROR","IWROR","WLBOR","WNAG","SNAG"]:
        register(algname, {}, cm.Paired(i/12))
        i += 2
    i = 1
    for algname in ["SSpiral"]:
        for spiral_turn_max in [3,6,10,18,30]:
            register(algname, {"spiral_turn_max":spiral_turn_max}, cm.gist_ncar(i/12))
            i += 2
    epm041_base("epm0417", 12, 6, 10)

def epm0418():
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
    epm041_base("epm0418", 12, 6, 10)




#TODO hessian miyou!




epm0418()