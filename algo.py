import numpy as np


def split_data_by_literal(data, item):
    data_pos, data_neg = [], []
    for x in data:
        if evaluate(item, x):
            data_pos.append(x)
        else:
            data_neg.append(x)
    return data_pos, data_neg


def evaluate(item, x):
    def _func(i, r, v):
        if i < -1:
            return _func(-i - 2, r, v) ^ 1
        if isinstance(v, str):
            if r == '==':
                return x[i] == v
            elif r == '!=':
                return x[i] != v
            else:
                return False
        elif isinstance(x[i], str):
            return False
        elif r == '<=':
            return x[i] <= v
        elif r == '>':
            return x[i] > v
        else:
            return False

    def _eval(i):
        if len(i) == 3:
            return _func(i[0], i[1], i[2])
        elif len(i) == 4:
            return evaluate(i, x)

    if len(item) == 0:
        return 0
    if len(item) == 3:
        return _func(item[0], item[1], item[2])
    if item[3] == 0 and len(item[1]) > 0 and not all([_eval(i) for i in item[1]]):
        return 0
    if len(item[2]) > 0 and any([_eval(i) for i in item[2]]):
        return 0
    return 1


def cover(item, x):
    return evaluate(item, x)


def classify(items, x):
    for i in items:
        if evaluate(i, x):
            return i[0][2]
    return None


def predict(rules, X):
    ret = []
    for x in X:
        ret.append(classify(rules, x))
    return ret


def ig(tp, fn, tn, fp):
    if tp + tn < fp + fn:
        return float('-inf')
    ret = 0
    tot_p, tot_n = float(tp + fp), float(tn + fn)
    tot = float(tot_p + tot_n)
    if tp > 0:
        ret += tp / tot * np.log(tp / tot_p)
    if fp > 0:
        ret += fp / tot * np.log(fp / tot_p)
    if tn > 0:
        ret += tn / tot * np.log(tn / tot_n)
    if fn > 0:
        ret += fn / tot * np.log(fn / tot_n)
    return ret


def best_ig(data_pos, data_neg, i, used_items=[]):
    xp, xn, cp, cn = 0, 0, 0, 0
    pos, neg = dict(), dict()
    xs, cs = set(), set()

    for d in data_pos:
        if pos.get(d[i]) is None:
            pos[d[i]], neg[d[i]] = 0, 0
        if isinstance(d[i], str):
            cs.add(d[i])
            pos[d[i]] += 1.0
            cp += 1.0
        else:
            xs.add(d[i])
            pos[d[i]] += 1.0
            xp += 1.0

    for d in data_neg:
        if neg.get(d[i]) is None:
            pos[d[i]], neg[d[i]] = 0, 0
        if isinstance(d[i], str):
            cs.add(d[i])
            neg[d[i]] += 1.0
            cn += 1.0
        else:
            xs.add(d[i])
            neg[d[i]] += 1.0
            xn += 1.0

    xs = list(xs)
    xs.sort()
    for j in range(1, len(xs)):
        pos[xs[j]] += pos[xs[j - 1]]
        neg[xs[j]] += neg[xs[j - 1]]

    best, v, r = float('-inf'), float('-inf'), ''

    for x in xs:
        if (i, '<=', x) not in used_items and (i, '>', x) not in used_items:
            ifg = ig(pos[x], xp - pos[x] + cp, xn - neg[x] + cn, neg[x])
            if best < ifg:
                best, v, r = ifg, x, '<='
            ifg = ig(xp - pos[x], pos[x] + cp, neg[x] + cn, xn - neg[x])
            if best < ifg:
                best, v, r = ifg, x, '>'

    for c in cs:
        if (i, '==', c) not in used_items and (i, '!=', c) not in used_items:
            ifg = ig(pos[c], cp - pos[c] + xp, cn - neg[c] + xn, neg[c])
            if best < ifg:
                best, v, r = ifg, c, '=='
            ifg = ig(cp - pos[c] + xp, pos[c], neg[c], cn - neg[c] + xn)
            if best < ifg:
                best, v, r = ifg, c, '!='
    return best, r, v


def best_feat(data_pos, data_neg, used_items=[]):
    if len(data_pos) == 0 and len(data_neg) == 0:
        return -1, '', ''
    n = len(data_pos[0]) if len(data_pos) > 0 else len(data_neg[0])
    _best = float('-inf')
    i, r, v = -1, '', ''
    for _i in range(n - 1):
        bg, _r, _v = best_ig(data_pos, data_neg, _i, used_items)
        if _best < bg:
            _best = bg
            i, r, v = _i, _r, _v
    return i, r, v


def majority(data, i=-1):
    tab = dict()
    for d in data:
        if tab.get(d[i]) is None:
            tab[d[i]] = 0
        tab[d[i]] += 1
    b, bn = '', 0
    for t in tab:
        if bn < tab[t]:
            bn, b = tab[t], t
    return i, '==', b


def foldrm(data, used_items=[], ratio=0.5):
    ret = []
    while len(data) > 0:
        item = majority(data, -1)
        data_pos, data_neg = split_data_by_literal(data, item)
        rule = learn_rule(data_pos, data_neg, used_items, ratio)
        tp = [i for i in range(len(data_pos)) if cover(rule, data_pos[i])]
        data = [data_pos[i] for i in range(len(data_pos)) if i not in set(tp)] + data_neg
        if len(tp) == 0:
            break
        rule = item, rule[1], rule[2], rule[3]
        ret.append(rule)
    return ret


def learn_rule(data_pos, data_neg, used_items=[], ratio=0.5):
    items = []
    flag = False
    while True:
        t = best_feat(data_pos, data_neg, used_items + items)
        items.append(t)
        rule = (-1, items, [], 0)
        data_tp = [data_pos[i] for i in range(len(data_pos)) if cover(rule, data_pos[i])]
        data_fp = [data_neg[i] for i in range(len(data_neg)) if cover(rule, data_neg[i])]
        if t[0] == -1 or len(data_fp) <= len(data_tp) * ratio:
            if t[0] == -1:
                items.pop()
                rule = (-1, items, [], 0)
            if len(data_fp) > 0 and t[0] != -1:
                flag = True
            break
        data_pos = data_tp
        data_neg = data_fp
    if flag:
        ab = fold(data_fp, data_tp, used_items + items, ratio)
        if len(ab) > 0:
            rule = (rule[0], rule[1], ab, 0)
    return rule


def fold(data_pos, data_neg, used_items=[], ratio=0.5):
    ret = []
    while len(data_pos) > 0:
        rule = learn_rule(data_pos, data_neg, used_items, ratio)
        tp = [i for i in range(len(data_pos)) if cover(rule, data_pos[i])]
        data_pos = [data_pos[i] for i in range(len(data_pos)) if i not in set(tp)]
        if len(tp) == 0:
            break
        ret.append(rule)
    return ret


def flatten_rules(rules):
    abrules = []
    ret = []
    rule_map = dict()
    flatten_rules.ab = -2

    def _eval(i):
        if isinstance(i, tuple) and len(i) == 3:
            return i
        elif isinstance(i, tuple):
            return _func(i)

    def _func(rule, root=False):
        t = (tuple(rule[1]), tuple([_eval(i) for i in rule[2]]))
        if t not in rule_map:
            rule_map[t] = rule[0] if root else flatten_rules.ab
            _ret = rule_map[t]
            if root:
                ret.append((_ret, t[0], t[1]))
            else:
                abrules.append((_ret, t[0], t[1]))
            if not root:
                flatten_rules.ab -= 1
        elif root:
            ret.append((rule[0], t[0], t[1]))
        return rule_map[t]

    for r in rules:
        _func(r, root=True)
    return ret + abrules


def add_constraint(rules):
    ret, abrules, rx = [], [], []
    k = 1
    for r in rules:
        if isinstance(r[0], tuple):
            prule = (k, r[1], r[2])
            crule = (r[0], (k,), tuple([i for i in range(1, k)]))
            ret.append(prule)
            rx.append(crule)
            k += 1
        else:
            abrules.append(r)
    rx.sort()
    return rx + ret + abrules


def justify(rs, x, idx=-1, pos=[]):
    for j in range(len(rs)):
        r = rs[j]
        i, d, ab = r[0], r[1], r[2]
        if idx == -1:
            pos.clear()
            if not isinstance(i, tuple):
                continue
            if not isinstance(d[0], tuple):
                if not all([justify(rs, x, idx=_j, pos=pos)[0] for _j in d]):
                    continue
            else:
                if not all([evaluate(_j, x) for _j in d]):
                    continue
        else:
            if i != idx:
                continue
            if not all([evaluate(_j, x) for _j in d]):
                continue
        if len(ab) > 0 and any([justify(rs, x, idx=_j, pos=pos)[0] for _j in ab]):
            continue
        if r not in pos:
            pos.append(r)
        if idx == -1:
            return i[2], j
        else:
            return 1, j
    if idx != -1:
        for r in rs:
            if r[0] == idx and r not in pos:
                pos.append(r)
    if idx == -1:
        return None, -1
    else:
        return 0, -1
