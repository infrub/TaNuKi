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
    def __init__(self, tensors):
        self.tensors = tensors
        self.length = len(tensors)
        self.troMemo = None
        self.contractor = None

    def generate_troMemo(self):
        if self.troMemo is not None:
            return self.troMemo

        unoMemo = {}
        binoMemo = {}
        for i in range(self.length):
            manB = 1 << i
            unoMemo[manB] = {"cost":0, "duct":self.tensors[i], "fatherB":None, "motherB":None, "expr":"args["+str(i)+"]"}
        def uno(manB):
            if manB in unoMemo:
                return unoMemo[manB]
            minre = {"cost":float("inf"), "duct":None, "fatherB":None, "motherB":None}
            for fatherB in inbits_iterator(manB):
                motherB = manB - fatherB
                re = bino(fatherB, motherB)
                if re["cost"] < minre["cost"]:
                    minre = {"cost":re["cost"], "duct":re["duct"], "fatherB":fatherB, "motherB":motherB}
            unoMemo[manB] = minre
            return minre
        def bino(fatherB, motherB):
            if (fatherB, motherB) in binoMemo:
                return binoMemo[(fatherB, motherB)]
            father = uno(fatherB)
            mother = uno(motherB)
            childDuct, birthCost, _ = tensorFrame_contract_common_and_cost(father["duct"], mother["duct"])
            re = {"cost":father["cost"]+mother["cost"]+birthCost, "duct":childDuct}
            binoMemo[(fatherB, motherB)] = re
            return re

        uno((1 << self.length)-1)

        troMemo = {}
        def tro(manB):
            if manB in troMemo:
                return troMemo[manB]
            re = uno(manB)
            if "expr" in re:
                pass
            else:
                expr = "(" + tro(re["fatherB"])["expr"] + "*" + tro(re["motherB"])["expr"] + ")"
                re.update({"expr":expr})
            troMemo[manB] = re
            return re

        tro((1 << self.length)-1)

        self.troMemo = troMemo
        return troMemo


    def generate_contractor(self):
        if self.contractor is not None:
            return self.contractor
        troMemo = self.generate_troMemo()
        f = eval("lambda *args: "+troMemo[(1 << self.length)-1]["expr"])
        self.contractor = f
        return f

    generate = generate_contractor

            


