import math


def split_data_by_item(data, item):
    data_pos, data_neg = [], []
    for x in data:
        if evaluate(item, x):
            data_pos.append(x)
        else:
            data_neg.append(x)
    return data_pos, data_neg


def evaluate(item, x):
    def __eval(i, r, v):
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
            return __eval(i[0], i[1], i[2])
        elif len(i) == 4:
            return evaluate(i, x)

    if len(item) == 0:
        return 0
    if len(item) == 3:
        return __eval(item[0], item[1], item[2])
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


def predict(rules, data):
    ret = []
    for x in data:
        ret.append(classify(rules, x))
    return ret


def gain(tp, fn, tn, fp):
    if tp + tn < fp + fn:
        return float('-inf')
    ret = 0
    tot_p, tot_n = float(tp + fp), float(tn + fn)
    tot = float(tot_p + tot_n)
    ret += tp / tot * math.log(tp / tot_p) if tp > 0 else 0
    ret += fp / tot * math.log(fp / tot_p) if fp > 0 else 0
    ret += tn / tot * math.log(tn / tot_n) if tn > 0 else 0
    ret += fn / tot * math.log(fn / tot_n) if fn > 0 else 0
    return ret


def best_ig(data_pos, data_neg, i, used_items=[]):
    xp, xn, cp, cn = 0, 0, 0, 0
    pos, neg = dict(), dict()
    xs, cs = set(), set()
    for d in data_pos:
        if d[i] not in pos:
            pos[d[i]], neg[d[i]] = 0, 0
        pos[d[i]] += 1.0
        if isinstance(d[i], str):
            cs.add(d[i])
            cp += 1.0
        else:
            xs.add(d[i])
            xp += 1.0
    for d in data_neg:
        if d[i] not in neg:
            pos[d[i]], neg[d[i]] = 0, 0
        neg[d[i]] += 1.0
        if isinstance(d[i], str):
            cs.add(d[i])
            cn += 1.0
        else:
            xs.add(d[i])
            xn += 1.0
    xs, cs = list(xs), list(cs)
    xs.sort()
    cs.sort()
    for j in range(1, len(xs)):
        pos[xs[j]] += pos[xs[j - 1]]
        neg[xs[j]] += neg[xs[j - 1]]
    best, v, r = float('-inf'), float('-inf'), ''
    for x in xs:
        if (i, '<=', x) in used_items or (i, '>', x) in used_items:
            continue
        ig = gain(pos[x], xp - pos[x] + cp, xn - neg[x] + cn, neg[x])
        if best < ig:
            best, v, r = ig, x, '<='
        ig = gain(xp - pos[x], pos[x] + cp, neg[x] + cn, xn - neg[x])
        if best < ig:
            best, v, r = ig, x, '>'
    for c in cs:
        if (i, '==', c) in used_items or (i, '!=', c) in used_items:
            continue
        ig = gain(pos[c], cp - pos[c] + xp, cn - neg[c] + xn, neg[c])
        if best < ig:
            best, v, r = ig, c, '=='
        ig = gain(cp - pos[c] + xp, pos[c], neg[c], cn - neg[c] + xn)
        if best < ig:
            best, v, r = ig, c, '!='
    return best, r, v


def best_item(X_pos, X_neg, used_items=[]):
    ret = -1, '', ''
    if len(X_pos) == 0 and len(X_neg) == 0:
        return ret
    n = len(X_pos[0]) if len(X_pos) > 0 else len(X_neg[0])
    best = float('-inf')
    for i in range(n - 1):
        ig, r, v = best_ig(X_pos, X_neg, i, used_items)
        if best < ig:
            best = ig
            ret = i, r, v
    return ret


def most(data, i=-1):
    tab = dict()
    for d in data:
        if d[i] not in tab:
            tab[d[i]] = 0
        tab[d[i]] += 1
    y, n = '', 0
    for t in tab:
        if n <= tab[t]:
            y, n = t, tab[t]
    return i, '==', y


def foldrm(data, ratio=0.5):
    ret = []
    while len(data) > 0:
        item = most(data)
        data_pos, data_neg = split_data_by_item(data, item)
        rule = learn_rule(data_pos, data_neg, [], ratio)
        data_fn = [data_pos[i] for i in range(len(data_pos)) if not cover(rule, data_pos[i])]
        if len(data_fn) == len(data_pos):
            break
        data = data_fn + data_neg
        rule = item, rule[1], rule[2], rule[3]
        ret.append(rule)
    return ret


def learn_rule(data_pos, data_neg, used_items=[], ratio=0.5):
    items = []
    while True:
        t = best_item(data_pos, data_neg, used_items + items)
        items.append(t)
        rule = -1, items, [], 0
        data_pos = [data_pos[i] for i in range(len(data_pos)) if cover(rule, data_pos[i])]
        data_neg = [data_neg[i] for i in range(len(data_neg)) if cover(rule, data_neg[i])]
        if t[0] == -1 or len(data_neg) <= len(data_pos) * ratio:
            if t[0] == -1:
                rule = -1, items[:-1], [], 0
            if len(data_neg) > 0 and t[0] != -1:
                ab = fold(data_neg, data_pos, used_items + items, ratio)
                if len(ab) > 0:
                    rule = rule[0], rule[1], ab, 0
            break
    return rule


def fold(data_pos, data_neg, used_items=[], ratio=0.5):
    ret = []
    while len(data_pos) > 0:
        rule = learn_rule(data_pos, data_neg, used_items, ratio)
        data_fn = [data_pos[i] for i in range(len(data_pos)) if not cover(rule, data_pos[i])]
        if len(data_pos) == len(data_fn):
            break
        data_pos = data_fn
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
