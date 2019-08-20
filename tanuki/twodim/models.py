import copy as copyModule
from tanuki import tensor_core as tnc
from tanuki import tensor_instant as tni
from tanuki import decomp as tnd
from tanuki.utils import *
from tanuki.errors import *
import textwrap
from math import sqrt
import logging
import numpy as np



class Fin2DSimTPS:
    def __init__(self, tensorss, phys_labelsss):
        self.tensors = [[tensor for tensor in tensors] for tensors in tensorss]
        self.phys_labelsss = [[phys_labels for phys_labels in phys_labelss] for phys_labelss in phys_labelsss]
