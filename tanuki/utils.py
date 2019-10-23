from collections import Counter
import random
import copy as copyModule
import math

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




pccs_table = {"v1":(185,31,87),
"v2":(208,47,72),
"v3":(221,68,59),
"v4":(233,91,35),
"v5":(230,120,0),
"v6":(244,157,0),
"v7":(241,181,0),
"v8":(238,201,0),
"v9":(210,193,0),
"v10":(168,187,0),
"v11":(88,169,29),
"v12":(0,161,90),
"v13":(0,146,110),
"v14":(0,133,127),
"v15":(0,116,136),
"v16":(0,112,155),
"v17":(0,96,156),
"v18":(0,91,165),
"v19":(26,84,165),
"v20":(83,74,160),
"v21":(112,63,150),
"v22":(129,55,138),
"v23":(143,46,124),
"v24":(173,46,108),
"b2":(239,108,112),
"b4":(250,129,85),
"b6":(255,173,54),
"b8":(250,216,49),
"b10":(183,200,43),
"b12":(65,184,121),
"b14":(0,170,159),
"b16":(0,152,185),
"b18":(41,129,192),
"b20":(117,116,188),
"b22":(161,101,168),
"b24":(208,103,142),
"s2":(197,63,77),
"s4":(204,87,46),
"s6":(225,146,21),
"s8":(222,188,3),
"s10":(156,173,0),
"s12":(0,143,86),
"s14":(0,130,124),
"s16":(0,111,146),
"s18":(0,91,155),
"s20":(83,76,152),
"s22":(124,61,132),
"s24":(163,60,106),
"dp2":(166,29,57),
"dp4":(171,61,29),
"dp6":(177,108,0),
"dp8":(179,147,0),
"dp10":(116,132,0),
"dp12":(0,114,67),
"dp14":(0,102,100),
"dp16":(0,84,118),
"dp18":(0,66,128),
"dp20":(62,51,123),
"dp22":(97,36,105),
"dp24":(134,29,85),
"lt+2":(241,152,150),
"lt+4":(255,167,135),
"lt+6":(255,190,113),
"lt+8":(242,217,110),
"lt+10":(199,211,109),
"lt+12":(133,206,158),
"lt+14":(98,192,181),
"lt+16":(91,175,196),
"lt+18":(108,154,197),
"lt+20":(144,145,195),
"lt+22":(176,136,181),
"lt+24":(217,142,165),
"lt2":(246,171,165),
"lt4":(255,185,158),
"lt6":(255,206,144),
"lt8":(251,230,143),
"lt10":(216,223,146),
"lt12":(156,217,172),
"lt14":(126,204,193),
"lt16":(121,186,202),
"lt18":(131,167,200),
"lt20":(162,159,199),
"lt22":(184,154,184),
"lt24":(218,160,179),
"sf2":(202,130,129),
"sf4":(218,146,122),
"sf6":(219,166,107),
"sf8":(211,189,108),
"sf10":(173,182,107),
"sf12":(118,177,138),
"sf14":(84,163,155),
"sf16":(81,146,164),
"sf18":(93,126,160),
"sf20":(120,120,160),
"sf22":(144,113,148),
"sf24":(180,120,139),
"d2":(163,90,92),
"d4":(175,105,84),
"d6":(179,127,70),
"d8":(171,148,70),
"d10":(133,143,70),
"d12":(79,135,102),
"d14":(42,123,118),
"d16":(36,106,125),
"d18":(52,89,125),
"d20":(84,82,124),
"d22":(108,74,113),
"d24":(139,79,101),
"dk2":(105,41,52),
"dk4":(117,54,42),
"dk6":(121,77,28),
"dk8":(116,96,31),
"dk10":(82,91,32),
"dk12":(35,82,58),
"dk14":(0,71,70),
"dk16":(0,69,88),
"dk18":(18,52,82),
"dk20":(50,45,81),
"dk22":(67,40,72),
"dk24":(97,45,70),
"p+2":(232,194,191),
"p+4":(235,194,181),
"p+6":(244,212,176),
"p+8":(242,230,184),
"p+10":(216,221,173),
"p+12":(174,212,185),
"p+14":(166,212,204),
"p+16":(173,209,218),
"p+18":(175,192,209),
"p+20":(187,189,208),
"p+22":(200,185,201),
"p+24":(222,196,202),
"p2":(231,213,212),
"p4":(233,213,207),
"p6":(246,227,206),
"p8":(239,230,198),
"p10":(230,233,198),
"p12":(196,224,203),
"p14":(191,224,217),
"p16":(198,221,226),
"p18":(194,204,213),
"p20":(201,202,213),
"p22":(208,200,209),
"p24":(228,213,217),
"ltg2":(192,171,170),
"ltg4":(193,171,165),
"ltg6":(206,187,168),
"ltg8":(198,190,161),
"ltg10":(189,193,162),
"ltg12":(157,182,165),
"ltg14":(152,182,177),
"ltg16":(158,180,185),
"ltg18":(155,165,175),
"ltg20":(162,162,175),
"ltg22":(171,160,171),
"ltg24":(189,172,176),
"g2":(116,92,92),
"g4":(117,92,87),
"g6":(128,108,92),
"g8":(120,111,87),
"g10":(110,114,90),
"g12":(83,102,90),
"g14":(78,103,100),
"g16":(79,101,108),
"g18":(76,87,101),
"g20":(86,85,102),
"g22":(96,82,98),
"g24":(114,92,99),
"dkg2":(62,45,48),
"dkg4":(63,46,44),
"dkg6":(74,60,50),
"dkg8":(68,62,48),
"dkg10":(61,64,51),
"dkg12":(42,52,46),
"dkg14":(39,52,52),
"dkg16":(39,52,57),
"dkg18":(34,41,51),
"dkg20":(41,39,52),
"dkg22":(48,37,49),
"dkg24":(61,46,52),
"Gy-9.5":(241,241,241),
"Gy-8.5":(214,214,214),
"Gy-7.5":(187,187,187),
"Gy-6.5":(161,161,161),
"Gy-5.5":(135,135,135),
"Gy-4.5":(109,109,109),
"Gy-3.5":(84,84,84),
"Gy-2.5":(60,60,60),
"Gy-1.5":(39,39,39),}

def pccs(moji):
    a = pccs_table.get(moji, (0,0,0))
    return (a[0]/255,a[1]/255,a[2]/255)