"""Phase 3 — fail until you implement leave_one_center_out."""
import numpy as np

from pancreasignal.data.urine import select_task
from pancreasignal.evaluation.splits import leave_one_center_out


def test_loco(fake_urine):
    rows, y = select_task(fake_urine, "pdac_vs_rest")
    folds = leave_one_center_out(rows["sample_origin"], y, min_positives=5)
    names = [f[0] for f in folds]
    assert names == ["AAA", "BBB", "CCC"], "TINY has only 2 PDAC -> must be skipped"
    for name, tr, te in folds:
        assert len(np.intersect1d(tr, te)) == 0
        assert (rows["sample_origin"].iloc[te] == name).all()
        assert (rows["sample_origin"].iloc[tr] != name).all()
        assert len(tr) + len(te) == len(rows)


def test_loco_skips_when_training_has_one_class():
    # All negatives live in center A (like controls at BPTB). Holding A out leaves only PDAC
    # to train on, so A must be skipped. B is all PDAC, so it is skipped too.
    import pandas as pd

    centers = pd.Series(["A"] * 10 + ["B"] * 10)
    y = pd.Series([0] * 5 + [1] * 5 + [1] * 10)
    assert leave_one_center_out(centers, y, min_positives=5) == []


def test_loco_uses_positions_not_index():
    # By POSITION: A has 6 PDAC (kept), B has 4 PDAC (< 5, skipped).
    # y's index runs backwards, so matching by index LABEL would swap A and B.
    import pandas as pd

    centers = pd.Series(["A"] * 10 + ["B"] * 10)
    y = pd.Series([0] * 4 + [1] * 6 + [0] * 6 + [1] * 4, index=range(19, -1, -1))
    folds = leave_one_center_out(centers, y, min_positives=5)
    assert [name for name, _, _ in folds] == ["A"]
