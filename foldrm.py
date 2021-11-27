from utils import load_data, split_data, get_scores, justify_data, decode_rules
from algo import foldrm, predict, classify, flatten_rules, justify, add_constraint


class Classifier:
    def __init__(self, attrs=None, numeric=None, label=None):
        self.attrs = attrs
        self.numeric = numeric
        self.label = label
        self.rules = None
        self.frs = None
        self.crs = None
        self.asp_rules = None
        self.seq = 1
        self.translation = None

    def load_data(self, file, amount=-1):
        data, self.attrs = load_data(file, self.attrs, self.label, self.numeric, amount)
        return data

    def fit(self, data, ratio=0.9):
        self.rules = foldrm(data, ratio=ratio)

    def predict(self, X):
        return predict(self.rules, X)

    def classify(self, x):
        return classify(self.rules, x)

    def asp(self):
        if self.asp_rules is None and self.rules is not None:
            self.frs = flatten_rules(self.rules)
            self.crs = add_constraint(self.frs)
            self.asp_rules = decode_rules(self.crs, self.attrs)
        return self.asp_rules

    def print_asp(self):
        for r in self.asp():
            print(r)

    def explain(self, x):
        pos = []
        justify(self.crs, x, pos=pos)
        expl = decode_rules(pos, self.attrs, x=x)
        for e in expl:
            print(e)
        print('')