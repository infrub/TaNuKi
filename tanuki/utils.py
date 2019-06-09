from collections import Counter

#utils
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


#label covering methods
#label :== string | tuple[label]
def normalize_argument_labels(labels):
    if isinstance(labels, list):
        return labels
    else:
        return [labels]

def normalize_and_complement_argument_labels(all_labels, row_labels, column_labels=None):
    row_labels = normalize_argument_labels(row_labels)
    if column_labels is None:
        column_labels = diff_list(all_labels, row_labels)
    else:
        column_labels = normalize_argument_labels(column_labels)
    return row_labels, column_labels

def unique_label():
    return "".join(random.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",k=8))

