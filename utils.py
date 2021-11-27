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


def get_scores(Y_hat, data):
    n = len(Y_hat)
    m = 0
    for i in range(n):
        if Y_hat[i] == data[i][-1]:
            m += 1
    return float(m) / n


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


def num_predicates(rules):
    def _n_pred(rule):
        return len(rule[1] + rule[2])
    ret = 0
    for r in rules:
        ret += _n_pred(r)
    return ret
