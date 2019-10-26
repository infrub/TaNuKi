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
from timeout_decorator import timeout, TimeoutError
from math import *

pd.options.display.max_columns = 30
pd.options.display.width = 160
np.set_printoptions(linewidth=float("inf"))
tnc.display_max_size = float("inf")



def epm0610():
    A = onedim.random_cyc1DBTPS([["p0"], ["p1"], ["p2"]], virt_labelss=[["v0"],["v1"],["v2"]])
    print(A.to_TMS())
    A.canonize()
    print(A.to_TMS())

epm0610()