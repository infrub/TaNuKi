import sys
sys.path.append('../../')
from tanuki import *
import numpy as np

def test0111():
    print(Tensor(np.arange(64).reshape(4,4,4),["a","b","c"])) 
    print(Tensor(np.arange(256).reshape(4,4,4,4),["a","b","c","d"]))

def test0112():
    try:
        print(Tensor(np.arange(256).reshape(4,4,4,4),["a","b","c","c"])) 
    except Exception as e:
        print(e)
    try:
        print(Tensor(np.arange(256).reshape(4,4,4,4),["a","b","c"])) 
    except Exception as e:
        print(e)

def test0113():
    a = Tensor(np.arange(12).reshape(3,4),["a","b"])
    a.move_index_to_top("b")
    print(a)

def test0114():
    a = Tensor(np.arange(12).reshape(3,4),["a","b"])
    b = a.move_index_to_top("b",inplace=False)
    print(b)
    print(a)

def test0115():
    a = Tensor(np.arange(30).reshape(2,3,5),["a","b","c"])
    b = a.move_index_to_position("c",1,inplace=False)
    print(b)
    c = a.move_index_to_position("a",2,inplace=False)
    print(c)

def test0116():
    a = Tensor(np.arange(120).reshape(2,3,4,5),["a","b","c","d"])
    b = a.move_indices_to_position(("a","c"),2,inplace=False)
    print(b)
    c = a.move_all_indices(["a","d","c","b"],inplace=False)
    print(c)

def test0117():
    a = Tensor(np.arange(120).reshape(2,3,4,5),["a","b","c","d"])
    print(a)
    info = a.fuse_indices(["d","b"])
    print(a)
    print(a.split_index(info["newLabelFuse"], info["oldShapeFuse"], inplace=False))
    a.split_index(info["newLabelFuse"], info["oldShapeFuse"], info["oldLabelsFuse"], inplace=True)
    print(a)


test0017()