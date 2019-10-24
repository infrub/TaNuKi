from collections import Counter
import random
import copy as copyModule
import math
import numpy as np

def soujou(xs):
    re = 1
    for x in xs:
        re *= x
    return re

#utils
def eq_list(xs, ys):
    xc = Counter(xs)
    yc = Counter(ys)
    xc.subtract(yc)
    zs = list(xc.elements())
    return len(zs)==0

def diff_list(univ, see):
    diff = list(univ)
    for x in see:
        diff.remove(x)
    return diff

def intersection_list(xs, ys):
    xc = Counter(xs)
    yc = Counter(ys)
    zc = xc & yc
    zs = list(zc.elements())
    return zs

def floor_half_list(xs):
    xc = Counter(xs)
    ys = []
    for x,c in xc.items():
        for _ in range(c // 2):
            ys.append(x)
    return ys

def indexs_duplable_front(univ, see):
    temp = list(univ)
    res = []
    for x in see:
        i = temp.index(x)
        res.append(i)
        temp[i] = None
    return res

def indexs_duplable_back(univ, see):
    temp = list(reversed(univ))
    res = []
    for x in see:
        i = temp.index(x)
        res.append(len(univ)-1-i)
        temp[i] = None
    return res

def more_popped_list(xs, xis):
    xs = list(xs)
    for xi in xis:
        xs[xi] = None
    xs = [x for x in xs if x is not None]
    return xs


#label covering methods
#label :== string | tuple[label]
def is_type_label(label):
    if isinstance(label, tuple):
        return all((is_type_label(x) for x in label))
    return isinstance(label, str)

def is_type_labels(labels):
    return all((is_type_label(x) for x in labels))

def normarg_labels(labels):
    if isinstance(labels, list):
        return labels
    else:
        return [labels]

def unique_label():
    return "".join(random.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",k=8))

def aster_label(label):
    if type(label)==tuple:
        return tuple(aster_label(x) for x in label)
    else:
        return label+"*"

def unaster_label(label):
    if type(label)==tuple:
        return tuple(unaster_label(x) for x in label)
    else:
        if label[-1] == "*":
            return label[:-1]
        else:
            return label

def prime_label(label):
    if type(label)==tuple:
        return tuple(prime_label(x) for x in label)
    else:
        return label+"'"

def unprime_label(label):
    if type(label)==tuple:
        return tuple(unaster_label(x) for x in label)
    else:
        if label[-1] == "'":
            return label[:-1]
        else:
            return label

def aster_labels(labels):
    return [aster_label(label) for label in labels]

def unaster_labels(labels):
    return [unaster_label(label) for label in labels]

def prime_labels(labels):
    return [prime_label(label) for label in labels]

def unprime_labels(labels):
    return [unprime_label(label) for label in labels]



class CyclicList(list):
    def __init__(self, *args, **kwargs):
        list.__init__(self, *args, **kwargs)
    def __repr__(self):
        return "CyclicList("+list.__repr__(self)+")"
    def __str__(self):
        return "CyclicList("+list.__repr__(self)+")"
    def __getitem__(self, i):
        return list.__getitem__(self, i%len(self))
    def __setitem__(self, i, v):
        return list.__setitem__(self, i%len(self), v)



class CollateralBool:
    def __init__(self, trueOrFalse, expression=None):
        self.trueOrFalse = bool(trueOrFalse)
        if expression is None: expression = {}
        self.expression = expression
    def __bool__(self):
        return self.trueOrFalse
    def __and__(x,y):
        return CollateralBool(x.trueOrFalse and y.trueOrFalse, {"op":"and", "left":x.expression, "right":y.expression})
    def __or__(x,y):
        return CollateralBool(x.trueOrFalse or y.trueOrFalse, {"op":"or", "left":x.expression, "right":y.expression})
    def __repr__(self):
        return f"{self.trueOrFalse}({self.expression})" #f"CollateralBool({self.trueOrFalse}, {self.expression})"
    def __str__(self):
        return f"{self.trueOrFalse}({self.expression})"
    def __getitem__(self, arg):
        return self.expression[arg]


class PowPowFloat: #[ (mantissa, radix, exponent) ]
    def __init__(self, *args):
        if len(args)==1:
            self.mres = args[0]
        else:
            self.mres = [(args[0],args[1],args[2])]
    def __str__(self):
        re = ""
        for m,r,e in self.mres:
            if len(re)>0:
                re += " * "
            re += f"({m})**({r}**{e})"
        return re
    def __mul__(self, other):
        if type(other) == PowPowFloat:
            return PowPowFloat(self.mres + other.mres)
        else:
            return self * PowPowFloat(other, self.mres[0][1], 0)
    @property
    def value(self):
        return soujou([m ** (r ** e) for (m,r,e) in self.mres])
    def __float__(self):
        result = self.value
        if type(result) == complex:
            result = result.real
        return result
    def __complex__(self):
        result = self.value
        if type(result) == float:
            result = result * (1.0+0.0j)
        return result
    @property
    def log(self, radix=math.e):
        result = 0
        for m,r,e in self.mres:
            result += math.log(abs(m),radix) * (r**e)
        return result


class ExpFloat:
    def __init__(self, *args):
        if len(args)==1:
            moto = args[0]
            if moto == 0:
                self.sign = 0
                self.ent = -float("inf")
            elif type(moto)==ExpFloat:
                self.sign = moto.sign
                self.ent = moto.ent
            elif type(moto)==complex:
                nrm = abs(moto)
                self.sign = moto / nrm
                self.ent = math.log(nrm)
            elif moto > 0:
                self.sign = 1
                self.ent = math.log(moto)
            else:
                self.sign = -1
                self.ent = math.log(-moto)
        elif len(args)==2:
            self.sign = args[0]
            self.ent = args[1]

    def __str__(self):
        if self.sign == 0:
            return f"0.0"
        elif type(self.sign)==complex:
            return f"({self.sign})*exp({self.ent})"
        elif self.sign == 1:
            return f"exp({self.ent})"
        else:
            return f"-exp({self.ent})"

    def __repr__(self):
        return f"ExpFloat({repr(self.sign)},{repr(self.ent)})"

    @property
    def value(self):
        return self.sign * math.exp(self.ent)

    @property
    def log(self):
        return self.ent

    @property
    def real(self):
        if type(self.sign)==complex:
            return ExpFloat(self.sign.real) * abs(self)
        else:
            return ExpFloat(self.sign) * abs(self)

    @property
    def imag(self):
        if type(self.sign)==complex:
            return ExpFloat(self.sign.imag) * abs(self)
        else:
            return ExpFloat(0.0)

    def __abs__(self):
        if self.sign == 0:
            return ExpFloat(0.0)
        else:
            return ExpFloat(1, self.ent)

    def __eq__(self, other):
        d = abs(self - other)
        return d.sign == 0 or d.ent < -15

    def __lt__(self, other):
        if type(other)==ExpFloat:
            if self.sign < other.sign:
                return True
            if self.sign > other.sign:
                return False
            if self.sign == 0:
                return False
            elif self.sign == 1:
                return self.ent < other.ent
            else:
                return self.ent > other.ent
        else:
            return self < ExpFloat(other)

    def __gt__(self, other):
        if type(other)==ExpFloat:
            if self.sign > other.sign:
                return True
            if self.sign < other.sign:
                return False
            if self.sign == 0:
                return False
            elif self.sign == 1:
                return self.ent > other.ent
            else:
                return self.ent < other.ent
        else:
            return self > ExpFloat(other)

    def __mul__(self, other):
        if type(other)==ExpFloat:
            return ExpFloat(self.sign * other.sign, self.ent + other.ent)
        else:
            return self * ExpFloat(other)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if type(other)==ExpFloat:
            return ExpFloat(self.sign / other.sign, self.ent - other.ent)
        else:
            return self / ExpFloat(other)

    def __neg__(self):
        return ExpFloat(-self.sign, self.ent)

    def __add__(self, other):
        if type(other)==ExpFloat:
            if type(self.sign)==complex or type(other.sign)==complex:
                return ef_complex(self.real+other.real, self.imag+other.imag)
            if other.sign == 0:
                return self
            elif self.sign == 0:
                return other
            elif self.sign == 1 and other.sign == -1:
                return self - (-other)
            elif self.sign == -1 and other.sign == 1:
                return other - (-self)
            elif self.sign == -1 and other.sign == -1:
                return -( (-self) + (-other) )

            # exp(a)+exp(b) = exp(a) * (1 + exp(b-a))
            # log(exp(a)+exp(b)) = a + log(1 + exp(b-a))
            if self < other:
                self,other = other,self
            a = self.ent
            b = other.ent
            c = a + np.log1p(math.exp(b-a))
            return ExpFloat(1,c)

        else:
            return self + ExpFloat(other)

    def __sub__(self, other):
        if type(other)==ExpFloat:
            if type(self.sign)==complex or type(other.sign)==complex:
                return ef_complex(self.real-other.real, self.imag-other.imag)
            if other.sign == 0:
                return self
            elif self.sign == 0:
                return -other
            elif self.sign == 1 and other.sign == -1:
                return self + (-other)
            elif self.sign == -1 and other.sign == 1:
                return -( (-self) + other )
            elif self.sign == -1 and other.sign == -1:
                return (-other) - (-self)

            if self < other:
                return -(other - self)

            # exp(a)-exp(b) = exp(a) * (1 - exp(b-a))
            # log(exp(a)-exp(b)) = a + log(1 - exp(b-a))
            a = self.ent
            b = other.ent
            if b-a > -1e-15:
                return ExpFloat(0.0)
            c = a + np.log1p(-math.exp(b-a))
            return ExpFloat(1,c)

        else:
            return self - ExpFloat(other)

    def __pow__(self, k):
        if self.sign==1:
            return ExpFloat(1, self.ent*k)
        elif self.sign==0:
            if k>0:
                return ExpFloat(0.0)
            elif k==0:
                return ExpFloat(1.0)
            else:
                raise ValueError
        else:
            if int(k) != k:
                raise ValueError
            else:
                return ExpFloat(self.sign**k, self.ent*k)

    def sqrt(self):
        if self.sign==1:
            return ExpFloat(1, self.ent/2)
        elif self.sign==0:
            return ExpFloat(0.0)
        else:
            raise ValueError


def ef_exp(k):
    return ExpFloat(1,k)

def ef_pow(r,k):
    return ExpFloat(r)**k

def ef_cosh(x):
    return ( ef_exp(x) + ef_exp(-x) ) / 2

def ef_sinh(x):
    return ( ef_exp(x) - ef_exp(-x) ) / 2

def ef_complex(x,y):
    if type(x) == ExpFloat and type(y) == ExpFloat:
        if x == 0 and y == 0:
            return ExpFloat(0.0)
        if x.ent >= y.ent:
            r = x.sign + y.sign * math.exp(y.ent-x.ent) * 1.0j
            r = ExpFloat(r)
            r.ent += x.ent
            return r
        else:
            r = x.sign * math.exp(x.ent-y.ent) + y.sign * 1.0j
            r = ExpFloat(r)
            r.ent += y.ent
            return r
    else:
        return ef_complex(ExpFloat(x),ExpFloat(y))



