import copy as copyModule
from tanuki import tensor_core as tnc
from tanuki.utils import *
from tanuki.errors import *
from tanuki.netcon.netcon_base import *
import textwrap
import random
import itertools
import logging
import sys



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
    




class NetconJermyn:
    def __init__(self, prime_tensors):
        self.prime_tensors = tensors_to_tensorFrames(prime_tensors)
        self.length = len(prime_tensors)
        self.eternity = None
        self.contractor = None

    def generate_eternity(self):
        if self.eternity is not None:
            return self.eternity

        tensordict_of_size = [{} for size in range(len(self.prime_tensors)+1)]
        tensordict_of_size[1] = {t.bits: t for t in self.prime_tensors}

        n = len(self.prime_tensors)
        xi_min = sys.float_info.max
        for t in self.prime_tensors:
            for x in t.shape:
                if x!=1:
                    xi_min = min(xi_min, x)

        mu_cap = 1.0
        prev_mu_cap = 0.0 #>=0

        while len(tensordict_of_size[-1])<1:
            logging.info("netcon: searching with mu_cap={0:.6e}".format(mu_cap))
            next_mu_cap = sys.float_info.max
            for c in range(2,n+1):
                for d1 in range(1,c//2+1):
                    d2 = c-d1
                    t1_t2_iterator = itertools.combinations(tensordict_of_size[d1].values(), 2) if d1==d2 else itertools.product(tensordict_of_size[d1].values(), tensordict_of_size[d2].values())
                    for t1, t2 in t1_t2_iterator:
                        if t1.is_overlap(t2): continue
                        if t2.is_disjoint(t2): continue

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

        self.eternity = tensordict_of_size[-1][(1<<n)-1]
        return self.eternity


    def generate_contractor(self):
        if self.contractor is not None:
            return self.contractor
        eternity = self.generate_eternity()
        f = eval("lambda *args: "+eternity.ifn)
        self.contractor = f
        return f

    generate = generate_contractor

            


