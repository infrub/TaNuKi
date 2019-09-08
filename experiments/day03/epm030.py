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

pd.options.display.max_columns = 30
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





def epm0302():
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

    #register("LBOR", {}, "gray")
    register("NOR", {}, "brown")
    omegas = np.linspace(1.05,1.95,19)
    for i,omega in enumerate(omegas):
        register("COR", {"omega":omega}, cm.gist_ncar((i+1)/(len(omegas)+1)))
    #register("ROR", {"omega_cands":[1.6,1.65,1.7,1.72,1.74,1.76,1.78,1.8,1.82,1.84,1.94,1.95]}, "black")

    metaf = open("epm0302_oups/epm0302_meta.csv","a")
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
        plt.suptitle(f"epm0302[b={b},chi={chi},seed={seed}]")
        plt.savefig(f"epm0302_oups/[b={b}, chi={chi}, seed={seed}].png", dpi=400)
        #plt.show()


    b=10
    for chi in range(1,10):
        for seed in range(10,30):
            print(f"b={b}, chi={chi}, seed={seed}")
            ikuze(b,chi,seed)
            print("\n")




def epm0303():
    df = pd.read_csv("epm0302_oups/epm0302_meta.csv", index_col=False)
    s = []
    for chi in range(1,10):
        s.append(df[df.chi==chi].mean())
    print(pd.DataFrame(s, index=list(range(1,10))))





def epm0304():
    registereds = []
    def register(algname, kwargs, color):
        if len(kwargs)==0:
            title = f"{algname}"
        else:
            title = f"{algname}("
            for kwarg in kwargs.values():
                if type(kwarg) in [float,np.float64]:
                    title += f"{kwarg:4.2f},"
                else:
                    title += f"{kwarg},"
            title = title[:-1] + ")"
        registereds.append( (algname,kwargs,color,title) )

    #register("LBOR", {}, "gray")
    register("NOR", {}, "brown")
    omegas = np.linspace(1.05,1.95,19)
    for i,omega in enumerate(omegas):
        register("COR", {"omega":omega}, cm.gist_ncar((i+1)/(len(omegas)+1)))
    #register("ROR", {"omega_cands":[1.6,1.65,1.7,1.72,1.74,1.76,1.78,1.8,1.82,1.84,1.94,1.95]}, "black")

    metaf = open("epm0304_oups/epm0304_meta.csv","a")
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

        #fig, axs = plt.subplots(1, 2, sharey=True, figsize=(13,8))
        #ax1,ax2 = tuple(axs)
        #ax1.set_xscale("log")
        #ax1.set_ylabel("precision scale")
        #ax1.set_xlabel("iteration times")
        #ax2.set_xlabel("times spent for getting over the precision range (lefter is better)")

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
                #ax1.plot(df.x, df.logmy, label=title, color=color)
                #ax2.plot(df.smoothed_tadasing_cost2, df.logmy, label=title, color=color) #migikara hidarini jikanha susumuyo
                print(f'{memo["iter_times"]}, {memo["sqdiff"]}')
                metaf.write(str(memo["iter_times"])+", ")
            except Exception as e:
                print(e)
                metaf.write("nan, ")

        metaf.write("\n")
        #ax2.set_xlim(0,min(100,ax2.get_xlim()[1]))
        #plt.legend()
        #plt.suptitle(f"epm0304[b={b},chi={chi},seed={seed}]")
        #plt.savefig(f"epm0304_oups/[b={b}, chi={chi}, seed={seed}].png", dpi=400)
        #plt.show()

    b=18
    for chi in range(1,18):
        for seed in range(30):
            print(f"b={b}, chi={chi}, seed={seed}")
            ikuze(b,chi,seed)
            print("\n")




def epm0305():
    big_df = pd.read_csv("epm0304_oups/epm0304_meta.csv", index_col=False)
    for chi in range(1,18):
        sr = big_df[big_df.chi==chi].mean()
        sr = sr.drop(["b","chi","seed"])
        print(chi, sr.idxmin())



def epm0306():
    registereds = []
    def register(algname, kwargs, color):
        if len(kwargs)==0:
            title = f"{algname}"
        else:
            title = f"{algname}("
            for kwarg in kwargs.values():
                if type(kwarg) in [float,np.float64]:
                    title += f"{kwarg:4.2f},"
                else:
                    title += f"{kwarg},"
            title = title[:-1] + ")"
        registereds.append( (algname,kwargs,color,title) )

    #register("LBOR", {}, "gray")
    register("NOR", {}, "brown")
    omegas = np.linspace(1.05,1.95,19)
    for i,omega in enumerate(omegas):
        register("COR", {"omega":omega}, cm.gist_ncar((i+1)/(len(omegas)+1)))
    #register("ROR", {"omega_cands":[1.6,1.65,1.7,1.72,1.74,1.76,1.78,1.8,1.82,1.84,1.94,1.95]}, "black")

    metaf = open("epm0306_oups/epm0306_meta.csv","a")
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

        #fig, axs = plt.subplots(1, 2, sharey=True, figsize=(13,8))
        #ax1,ax2 = tuple(axs)
        #ax1.set_xscale("log")
        #ax1.set_ylabel("precision scale")
        #ax1.set_xlabel("iteration times")
        #ax2.set_xlabel("times spent for getting over the precision range (lefter is better)")

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
                #ax1.plot(df.x, df.logmy, label=title, color=color)
                #ax2.plot(df.smoothed_tadasing_cost2, df.logmy, label=title, color=color) #migikara hidarini jikanha susumuyo
                print(f'{memo["iter_times"]}, {memo["sqdiff"]}')
                metaf.write(str(memo["iter_times"])+", ")
            except Exception as e:
                print(e)
                metaf.write("nan, ")

        metaf.write("\n")
        #ax2.set_xlim(0,min(100,ax2.get_xlim()[1]))
        #plt.legend()
        #plt.suptitle(f"epm0306[b={b},chi={chi},seed={seed}]")
        #plt.savefig(f"epm0306_oups/[b={b}, chi={chi}, seed={seed}].png", dpi=400)
        #plt.show()

    b=6
    for chi in range(1,b):
        for seed in range(30):
            print(f"b={b}, chi={chi}, seed={seed}")
            ikuze(b,chi,seed)
            print("\n")



def epm0307():
    big_df = pd.read_csv("epm0306_oups/epm0306_meta.csv", index_col=False)
    for chi in range(1,6):
        sr = big_df[big_df.chi==chi].mean()
        sr = sr.drop(["b","chi","seed"])
        print(chi, sr.idxmin())




def epm0308():
    plt.figure()
    plt.xlabel("x")
    plt.ylabel("y")
    for (b,epmName) in [(6,"epm0306"),(10,"epm0302"),(18,"epm0304")]:
        big_df = pd.read_csv(f"{epmName}_oups/{epmName}_meta.csv", index_col=False)
        chis = list(range(1,b))
        bestOmegas = []
        for chi in chis:
            sr = big_df[big_df.chi==chi].mean()
            sr = sr.drop(["b","chi","seed"])
            bestOmega = sr.idxmin()
            bestOmega = float(bestOmega[4:-1])
            bestOmegas.append(bestOmega)
            print(f"{b}, {chi}: {bestOmega}")
        print()
        chis = np.array(chis)
        bestOmegas = np.array(bestOmegas)
        bo_c1s = 1/(2-bestOmegas)
        #plt.plot(chis*b, bo_c1s, label=f"b={b}")
        plt.plot(1/(chis*b), 2-bestOmegas, label=f"b={b}")
        #plt.plot(np.log(chis)+np.log(b), bestOmegas, label=f"b={b}")
    def f(x):
        return 0.125*np.log(x)+0.8
    plt.plot(np.linspace(0.002,0.2,1000), f(np.linspace(0.002,0.2,1000)), label=f"f")
    plt.xscale("log")
    plt.legend()
    plt.show()

    # tanjun ni chi ni izon?


if __name__ == "__main__":
    epm0308()