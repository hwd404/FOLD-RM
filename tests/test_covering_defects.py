"""Regression tests for two defects in sequential covering (algo.foldrm).

Fixture: 18 rows of real data (British bird species; AVONET / EltonTraits-derived
features, trophic niche as the label), reduced from a 581-row table by delta
debugging. Run with:  python -m pytest tests/   (or just: python tests/test_covering_defects.py)
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import algo  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CATEGORICAL = ['family', 'habitat', 'migration', 'primary_lifestyle']
NUMERIC = ['body_mass_g', 'beak_depth_mm']
LABEL = 'trophic_niche'


def load():
    with open(os.path.join(HERE, 'birds_covering_defect.csv'), newline='') as f:
        rows = list(csv.DictReader(f))
    return [[r[c] for c in CATEGORICAL] + [float(r[c]) for c in NUMERIC] + [r[LABEL]] for r in rows]


def coverage(rules, data):
    return sum(any(algo.cover(r, d) for r in rules) for d in data) / len(data)


def test_every_row_is_covered():
    data = load()
    rules = algo.foldrm(list(data))
    assert coverage(rules, data) == 1.0


def test_every_class_gets_a_rule():
    data = load()
    rules = algo.foldrm(list(data))
    assert {r[0][2] for r in rules} == {d[-1] for d in data}


def test_gain_rejects_literals_covering_no_positives():
    # tp == 0 but would otherwise score: it excludes 9 of 10 negatives.
    assert algo.gain(0, 5, 9, 1) == float('-inf')


def test_gain_unchanged_when_positives_covered():
    assert algo.gain(3, 2, 9, 1) > float('-inf')


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            try:
                fn()
                print('PASS', name)
            except AssertionError:
                print('FAIL', name)
