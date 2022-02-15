import random
from algo import evaluate, justify


def load_data(file, attrs, label, numerics, amount=-1):
    f = open(file, 'r')
    attr_idx, num_idx, lab_idx = [], [], -1
    ret, i, k = [], 0, 0
    head = ''
    for line in f.readlines():
        if i == 0:
            line = line.strip('\n').split(',')
            attr_idx = [j for j in range(len(line)) if line[j] in attrs]
            num_idx = [j for j in range(len(line)) if line[j] in numerics]
            for j in range(len(line)):
                if line[j] == label:
                    lab_idx = j
                    head += line[j]
        else:
            line = line.strip('\n').split(',')
            r = [j for j in range(len(line))]
            for j in range(len(line)):
                if j in num_idx:
                    try:
                        r[j] = float(line[j])
                    except:
                        r[j] = line[j]
                else:
                    r[j] = line[j]
            r = [r[j] for j in attr_idx]
            if lab_idx != -1:
                y = line[lab_idx]
                r.append(y)
            ret.append(r)
        i += 1
        amount -= 1
        if amount == 0:
            break
    attrs.append(head)
    return ret, attrs


def split_data(data, ratio=0.8, shuffle=True):
    if shuffle:
        random.shuffle(data)
    num = int(len(data) * ratio)
    train, test = data[: num], data[num:]
    return train, test


def over_sample(data, each=-1):
    tab = {}
    for d in data:
        y = d[-1]
        if y not in tab:
            tab[y] = []
        tab[y].append(d)
    ret, n = [], 0
    for t in tab:
        n = max(n, len(tab[t]))
    n = each if each > 0 else n
    for t in tab:
        d = tab[t] * int(n / len(tab[t]) + 1)
        random.shuffle(d)
        ret.extend(d[:n])
    tmp = [d[-1] for d in ret]
    tab = {}
    for t in tmp:
        if t not in tab:
            tab[t] = 0
        tab[t] += 1
    print('% over sample size', len(ret), tab)
    return ret


def get_scores(Y_hat, data):
    n = len(Y_hat)
    m = 0
    for i in range(n):
        if Y_hat[i] == data[i][-1]:
            m += 1
    return float(m) / n


def scores(Y_hat, Y, weighted=False):
    n = len(Y_hat)
    tp, fp, fn = {}, {}, {}
    for i in range(n):
        y, yh = Y[i], Y_hat[i]
        if y not in tp:
            tp[y], fp[y], fn[y] = 0, 0, 0
        if yh not in tp:
            tp[yh], fp[yh], fn[yh] = 0, 0, 0
        if yh == y:
            tp[y] += 1
        else:
            fp[yh] += 1
            fn[y] += 1
    p_mic = float(sum([tp[y] for y in tp])) / sum([tp[y] + fp[y] for y in tp])
    if weighted:
        p_mac = 1.0 / n * sum([float(tp[y]) * (tp[y] + fn[y]) / (tp[y] + fp[y]) for y in tp if tp[y] + fp[y] > 0])
        r_mac = 1.0 / n * sum([float(tp[y]) * (tp[y] + fn[y]) / (tp[y] + fn[y]) for y in tp if tp[y] + fn[y] > 0])
        f1_mac = 2.0 / n * sum([(tp[y] + fn[y]) * (float(tp[y]) / (tp[y] + fp[y]) * float(tp[y]) / (tp[y] + fn[y]))
                                / (float(tp[y]) / (tp[y] + fp[y]) + float(tp[y]) / (tp[y] + fn[y])) for y in tp
                                if tp[y] + fn[y] > 0 and tp[y] + fp[y] > 0 and tp[y] > 0])
    else:
        p_mac = 1.0 / len(tp) * sum([float(tp[y]) / (tp[y] + fp[y]) for y in tp if tp[y] + fp[y] > 0])
        r_mac = 1.0 / len(tp) * sum([float(tp[y]) / (tp[y] + fn[y]) for y in tp if tp[y] + fn[y] > 0])
        f1_mac = 2 * (p_mac * r_mac) / (p_mac + r_mac) if p_mac + r_mac > 0 else 0
    return p_mic, p_mac, r_mac, f1_mac


def justify_data(frs, x, attrs):
    ret = []
    for r in frs:
        d = r[1]
        if isinstance(d[0], tuple):
            for j in d:
                ret.append(attrs[j[0]] + ': ' + str(x[j[0]]))
    return set(ret)


def decode_rules(rules, attrs, x=None):
    ret = []
    nr = {'<=': '>', '>': '<=', '==': '!=', '!=': '=='}

    def _f1(it):
        prefix, not_prefix = '', ''
        if isinstance(it, tuple) and len(it) == 3:
            if x is not None:
                if it[0] == -1:
                    prefix = '[T]' if justify(rules, x)[0] == it[2] else '[F]'
                else:
                    prefix = '[T]' if evaluate(it, x) else '[F]'
                not_prefix = '[T]' if prefix == '[F]' else '[F]'
            i, r, v = it[0], it[1], it[2]
            if i < -1:
                i = -i - 2
                r = nr[r]
            k = attrs[i].lower().replace(' ', '_')
            if isinstance(v, str):
                v = v.lower().replace(' ', '_')
                v = 'null' if len(v) == 0 else '\'' + v + '\''
            if r == '==':
                return prefix + k + '(X,' + v + ')'
            elif r == '!=':
                return 'not ' + not_prefix + k + '(X,' + v + ')'
            else:
                return prefix + k + '(X,' + 'N' + str(i) + ')' + ', N' + str(i) + r + str(round(v, 3))
        elif it == -1:
            pass
        else:
            if x is not None:
                if it not in [r[0] for r in rules]:
                    prefix = '[U]'
                else:
                    prefix = '[T]' if justify(rules, x, it)[0] else '[F]'
                    pass
            if it < -1:
                return prefix + 'ab' + str(abs(it) - 1) + '(X)'
            else:
                return prefix + 'rule' + str(abs(it)) + '(X)'

    def _f2(rule):
        head = _f1(rule[0])
        body = ''
        for i in list(rule[1]):
            body = body + _f1(i) + ', '
        tail = ''
        for i in list(rule[2]):
            t = _f1(i)
            if 'not' not in t:
                tail = tail + 'not ' + _f1(i) + ', '
            else:
                t = t.replace('not ', '')
                tail = tail + t + ', '
        _ret = head + ' :- ' + body + tail
        chars = list(_ret)
        chars[-2] = '.'
        _ret = ''.join(chars)
        _ret = _ret.replace('<=', '=<')
        return _ret

    for _r in rules:
        ret.append(_f2(_r))
    return ret


def zip_rule(rule):
    tab, dft = {}, []
    for i in rule[1]:
        if isinstance(i[2], str):
            dft.append(i)
        else:
            if i[0] not in tab:
                tab[i[0]] = []
            if i[1] == '<=':
                tab[i[0]].append([float('-inf'), i[2]])
            else:
                tab[i[0]].append([i[2], float('inf')])
    nums = [t for t in tab]
    nums.sort()
    for i in nums:
        left, right = float('inf'), float('-inf')
        for j in tab[i]:
            if j[0] == float('-inf'):
                left = min(left, j[1])
            else:
                right = max(right, j[0])
        if left == float('inf'):
            dft.append((i, '>', right))
        elif right == float('-inf'):
            dft.append((i, '<=', left))
        else:
            dft.append((i, '>', right))
            dft.append((i, '<=', left))
    return rule[0], dft, rule[2], 0


def simplify_rule(rule):
    head, body = rule.split(' :- ')
    items = body.split(', ')
    items = list(dict.fromkeys(items))
    body = ', '.join(items)
    return head + ' :- ' + body


def num_predicates(rules):
    def _n_pred(rule):
        return len(rule[1] + rule[2])
    ret = 0
    for r in rules:
        ret += _n_pred(r)
    return ret


def fitem(rules, attrs, x, it):
    nr = {'<=': '>', '>': '<=', '==': '!=', '!=': '=='}
    if isinstance(it, tuple) and len(it) == 3 and not isinstance(it[2], tuple) and it[0] != -1:
        suffix = ' (DOES HOLD) ' if evaluate(it, x) else ' (DOES NOT HOLD) '
        i, r, v = it[0], it[1], it[2]
        if i < -1:
            i = -2 - i
            r = nr[r]
        k = attrs[i].lower().replace(' ', '_')
        if isinstance(v, str):
            v = v.lower().replace(' ', '_')
            v = 'null' if len(v) == 0 else '\'' + v + '\''
        xi = x[i]
        if isinstance(xi, str):
            xi = xi.lower().replace(' ', '_')
            xi = '\'null\'' if len(xi) == 0 else xi
        if r == '==':
            return 'the value of ' + k + ' is \'' + str(xi) + '\' which should equal ' + v + suffix
        elif r == '!=':
            return 'the value of ' + k + ' is \'' + str(xi) + '\' which should not equal ' + v + suffix
        else:
            if r == '<=':
                return 'the value of ' + k + ' is ' + str(xi) + ' which should be less equal to ' + str(round(v, 3)) + suffix
            else:
                return 'the value of ' + k + ' is ' + str(xi) + ' which should be greater than ' + str(round(v, 3)) + suffix
    elif isinstance(it, tuple) and len(it) == 3 and not isinstance(it[2], tuple) and it[0] == -1:
        suffix = ' DOES HOLD ' if justify(rules, x)[0] else ' DOES NOT HOLD '
        return 'the value of ' + attrs[-1] + ' is ' + str(it[2]) + suffix
    else:
        if it not in [r[0] for r in rules]:
            pass
        elif it < -1:
            suffix = ' DOES HOLD ' if justify(rules, x, it)[0] else ' DOES NOT HOLD '
            return 'exception ab' + str(abs(it) - 1) + suffix
        else:
            suffix = ' DOES HOLD ' if justify(rules, x, it)[0] else ' DOES NOT HOLD '
            return 'rule' + str(it) + suffix


def frules(rules, attrs, x, rule, indent=0):
    head = '\t' * indent + fitem(rules, attrs, x, rule[0]) + 'because \n'
    body = ''
    if not isinstance(rule[0], tuple):
        for i in list(rule[1]):
            body = body + '\t' * (indent + 1) + fitem(rules, attrs, x, i) + '\n'
    else:
        for i in list(rule[1]):
            if isinstance(i, tuple):
                body = body + '\t' * (indent + 1) + fitem(rules, attrs, x, i) + '\n'
            else:
                for r in rules:
                    if i == r[0]:
                        body = body + frules(rules, attrs, x, r, indent + 1)
    tail = ''
    for i in list(rule[2]):
        for r in rules:
            if i == r[0]:
                tail = tail + frules(rules, attrs, x, r, indent + 1)
    _ret = head + body + tail
    chars = list(_ret)
    _ret = ''.join(chars)
    return _ret


def proof_tree(rules, attrs, x):
    ret = []
    for r in rules:
        if isinstance(r[0], tuple):
            ret.append(frules(rules, attrs, x, r))
    return ret
