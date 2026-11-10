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
