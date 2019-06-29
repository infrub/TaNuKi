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
        self.prime_tensors = tensors_to_tensorFrames(prime_tensors)
        self.length = len(prime_tensors)
        self.eternity = None
        self.contractor = None

    def generate_eternity(self):
        if self.eternity is not None:
            return self.eternity

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
                child = tensorFrame_contract_common(uno(fatherB), uno(motherB))
                if minChild is None or child.cost < minChild.cost:
                    minChild = child
            unoMemo[manB] = minChild
            return minChild

        #uno((1 << self.length)-1)
        #for k,v in unoMemo.items():
        #    print(k,v)
        self.eternity = uno((1 << self.length)-1)
        return self.eternity


    def generate_contractor(self):
        if self.contractor is not None:
            return self.contractor
        eternity = self.generate_eternity()
        f = eval("lambda *args: "+eternity.ifn)
        self.contractor = f
        return f

    generate = generate_contractor

            


