import copy as copyModule
from tanuki import tensor_core as tnc
from tanuki.utils import *
from tanuki.errors import *
from tanuki.netcon.netcon_base import *
import textwrap
import random



def inbits_iterator(z):
    """
    y = 0
    yield y
    while y!=z:
        y = (y-z)&z
        yield y
    return
    """
    y = (-z)&z
    while y!=z:
        yield y
        y = (y-z)&z
    return
    




class NetconBrute:
    def __init__(self, prime_tensors):
        self.prime_tensors = prime_tensors
        self.length = len(prime_tensors)
        self.root_child = None
        self.contractor = None

    def generate_root_child(self):
        if self.root_child is not None:
            return self.root_child

        unoMemo = {}
        for i in range(self.length):
            manB = 1 << i
            unoMemo[manB] = self.prime_tensors[i]
        def uno(manB):
            if manB in unoMemo:
                return unoMemo[manB]
            minChild = None
            for fatherB in inbits_iterator(manB):
                motherB = manB - fatherB
                child = tensorFrame_contract_common_and_cost(uno(fatherB), uno(motherB))
                if minChild is None or child.cost < minChild.cost:
                    minChild = child
            unoMemo[manB] = minChild
            return minChild

        #uno((1 << self.length)-1)
        #for k,v in unoMemo.items():
        #    print(k,v)
        return uno((1 << self.length)-1)


    def generate_contractor(self):
        if self.contractor is not None:
            return self.contractor
        root_child = self.generate_root_child()
        f = eval("lambda *args: "+root_child.ifn)
        self.contractor = f
        return f

    generate = generate_contractor

            


