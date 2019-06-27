from collections import Counter
import random


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
    def __getitem__(self, i):
        return list.__getitem__(self, i%len(self))
    def __setitem__(self, i, v):
        return list.__setitem__(self, i%len(self), v)



class CollateralBool:
    def __init__(self, trueOrFalse, expression):
        self.trueOrFalse = bool(trueOrFalse)
        self.expression = expression
    def __bool__(self):
        return self.trueOrFalse
    def __repr__(self):
        return f"CollateralBool({self.trueOrFalse}, {self.expression})"
    def __str__(self):
        return f"{self.trueOrFalse}({self.expression})"
    def __getitem__(self, arg):
        return self.expression[arg]