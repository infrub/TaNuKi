import copy as copyModule
from tanuki import tensor_core as tnc
from tanuki.utils import *
from tanuki.errors import *
import textwrap
import sys
import itertools
import logging



class TensorFrame(tnc.TensorLabelingMixin):
    def __init__(self, shape, labels, *args):
        self.shape = shape
        self.ndim = len(shape)
        self.labels = labels
        if len(args)==0:
            pass
        elif len(args)==1:
            prime_id = args[0]
            self.bits = (1<<prime_id)
            self.rpn = [prime_id]
            self.ifn = f"args[{prime_id}]"
            self.cost = 0.0
        else:
            self.bits = args[0]
            self.rpn = args[1]
            self.ifn = args[2]
            if len(args)==4:
                self.cost = float(args[3])
            else:
                self.cost = 0.0
        self.is_new = True

    def __repr__(self):
        return f"TensorFrame(shape={self.shape}, labels={self.labels}, rpn={self.rpn}, ifn={self.ifn}, cost={self.cost})"

    def __str__(self):
        re = \
        f"TensorFrame(\n" + \
        f"    shape={self.shape},\n" + \
        f"    labels={self.labels},\n" + \
        f"    rpn={self.rpn},\n" + \
        f"    ifn={self.ifn},\n" + \
        f"    cost={self.cost},\n" + \
        f")"
        return re

    def __mul__(self, other):
        return tensorFrame_contract_common(self, other)

    def is_overlap(self, other):
        return (self.bits & other.bits)>0

    def is_disjoint(self, other):
        return len(intersection_list(self.labels, other.labels))==0



def tensorFrame_contract_common(A, B):
    commonLabels = intersection_list(A.labels, B.labels)
    aIndicesContract, aIndicesNotContract = A.normarg_complement_indices_back(commonLabels)
    bIndicesContract, bIndicesNotContract = B.normarg_complement_indices_front(commonLabels)
    aLabelsContract, aLabelsNotContract = A.labels_of_indices(aIndicesContract), A.labels_of_indices(aIndicesNotContract)
    bLabelsContract, bLabelsNotContract = B.labels_of_indices(bIndicesContract), B.labels_of_indices(bIndicesNotContract)
    aDimsContract, aDimsNotContract = A.dims(aIndicesContract), A.dims(aIndicesNotContract)
    bDimsContract, bDimsNotContract = B.dims(bIndicesContract), B.dims(bIndicesNotContract)
    assert aDimsContract == bDimsContract
    cDims = aDimsNotContract + bDimsNotContract
    cLabels = aLabelsNotContract + bLabelsNotContract
    cBits = A.bits + B.bits
    cRpn = A.rpn + B.rpn + ["*"]
    cIfn = "(" + A.ifn + "*" + B.ifn +")"
    elim = soujou(aDimsContract)
    cCost = A.cost + B.cost + soujou(cDims)*elim
    return TensorFrame(cDims, cLabels, cBits, cRpn, cIfn, cCost)

def tensors_to_tensorFrames(Ts):
    TFs = []
    for i,T in enumerate(Ts):
        TF = TensorFrame(T.shape, T.labels, i)
        TFs.append(TF)
    return TFs





class AutoContractor:
    def __init__(self, primeTs=None):
        if primeTs is None:
            self.primeTs = None
            self.primeTFs = None
            self.length = None
            self.eternity = None
        else:
            self.primeTs = primeTs
            self.primeTFs = tensors_to_tensorFrames(primeTs)
            self.length = len(primeTs)
            self.eternity = None


    def __repr__(self):
        return "AutoContractor(" + repr(self.primeTs) + ")"

    def __str__(self):
        if self.primeTFs is None:
            dataStr = "undefined\n"
        else:
            dataStr = ""
            for tf in self.primeTFs:
                dataStr += str(tf) + ",\n"
            dataStr = textwrap.indent(dataStr, "    ")
            dataStr = "[\n" + dataStr + "]\n"
        if self.eternity is None:
            dataStr += "rpn: undefined\n"
            dataStr += "cost: undefined\n"
        else:
            dataStr += f"rpn: {self.eternity.rpn}\n"
            dataStr += f"cost: {self.eternity.cost}\n"
        dataStr = textwrap.indent(dataStr, "    ")
        dataStr = "AutoContractor(\n" + dataStr + ")"
        return dataStr

    @property
    def rpn(self): return self.eternity.rpn
    @property
    def cost(self): return self.eternity.cost

    def set_primeTs(self, primeTs, same_frame=True):
        self.primeTs = primeTs
        if self.primeTFs is None or not same_frame:
            self.primeTFs = tensors_to_tensorFrames(primeTs)
            self.length = len(primeTs)
            self.eternity = None
    

    def _generate_eternity_brute(self):
        def inbits_iterator(z):
            y = (-z)&z
            while y!=z:
                yield y
                y = (y-z)&z
            return

        unoMemo = {}
        for i in range(self.length):
            manB = 1 << i
            unoMemo[manB] = self.primeTFs[i]

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

        return uno((1 << self.length)-1)


    # ref: https://github.com/smorita/Tensordot
    def _generate_eternity_jermyn(self):
        tensordict_of_size = [{} for size in range(len(self.primeTFs)+1)]
        tensordict_of_size[1] = {t.bits: t for t in self.primeTFs}

        n = len(self.primeTFs)
        xi_min = sys.float_info.max
        for t in self.primeTFs:
            for x in t.shape:
                if x!=1:
                    xi_min = min(xi_min, x)

        mu_cap = 1.0
        prev_mu_cap = 0.0

        while len(tensordict_of_size[-1])<1:
            logging.info("netcon: searching with mu_cap={0:.6e}".format(mu_cap))
            next_mu_cap = sys.float_info.max
            for c in range(2,n+1):
                for d1 in range(1,c//2+1):
                    d2 = c-d1
                    t1_t2_iterator = itertools.combinations(tensordict_of_size[d1].values(), 2) if d1==d2 else itertools.product(tensordict_of_size[d1].values(), tensordict_of_size[d2].values())
                    for t1, t2 in t1_t2_iterator:
                        if t1.is_overlap(t2): continue
                        if t1.is_disjoint(t2): continue

                        t_new = tensorFrame_contract_common(t1,t2)

                        if next_mu_cap <= t_new.cost:
                            pass
                        elif mu_cap < t_new.cost:
                            next_mu_cap = t_new.cost
                        elif t1.is_new or t2.is_new or prev_mu_cap < t_new.cost:
                            t_old = tensordict_of_size[c].get(t_new.bits)
                            if t_old is None or t_new.cost < t_old.cost:
                                tensordict_of_size[c][t_new.bits] = t_new
            prev_mu_cap = mu_cap
            mu_cap = max(next_mu_cap, mu_cap*xi_min)
            for s in tensordict_of_size:
                for t in s.values(): t.is_new = False

            logging.debug("netcon: tensor_num=" +  str([ len(s) for s in tensordict_of_size]))

        return tensordict_of_size[-1][(1<<n)-1]


    def _generate_eternity(self, algorithm="Jermyn"):
        if algorithm=="Jermyn":
            return self._generate_eternity_jermyn()
        elif algorithm=="Brute":
            return self._generate_eternity_brute()
        else:
            raise ValueError


    def get_eternity(self):
        if self.eternity is None:
            self.eternity = self._generate_eternity()
        return self.eternity


    def exec(self, primeTs=None, same_frame=True):
        if primeTs is not None:
            self.set_primeTs(primeTs, same_frame=same_frame)
        rpn = self.get_eternity().rpn
        stack = []
        for c in rpn:
            if c=="*":
                father = stack.pop()
                mother = stack.pop()
                child = father * mother
                stack.append(child)
            else:
                stack.append(self.primeTs[c])
        return stack[0]
        