import numpy as np

from mpcg_wav2vec.datasets.splits import SplitRatios, make_splits, read_cinc_labels
from mpcg_wav2vec.reporting import flatten_metrics, summarize, to_markdown


def _labels(n=100):
    # balanced-ish binary labels in CinC {-1, 1} encoding
    return {f"a{i:04d}": (1 if i % 2 == 0 else -1) for i in range(n)}


def test_make_splits_columns_and_partition():
    df = make_splits(_labels(100), folds=3, ratios=SplitRatios(0.6, 0.2, 0.2), seed=1)
    assert list(df.columns) == ["patient", "label", "split", "split2", "split3"]
    assert len(df) == 100
    for col in ("split", "split2", "split3"):
        vals = set(df[col])
        assert vals <= {"train", "valid", "test"}
        # roughly 60/20/20
        frac_test = (df[col] == "test").mean()
        assert 0.1 < frac_test < 0.35


def test_make_splits_deterministic_and_stratified():
    df1 = make_splits(_labels(80), folds=2, seed=7)
    df2 = make_splits(_labels(80), folds=2, seed=7)
    assert df1.equals(df2)
    # both classes present in every subset of fold 1
    for subset in ("train", "valid", "test"):
        labels_in = set(df1[df1["split"] == subset]["label"])
        assert labels_in == {-1, 1}


def test_read_cinc_labels(tmp_path):
    (tmp_path / "REFERENCE.csv").write_text("a0001,1\na0002,-1\na0003,1\n")
    labels = read_cinc_labels(str(tmp_path))
    assert labels == {"a0001": 1, "a0002": -1, "a0003": 1}


def test_reporting_flatten_and_summarize():
    records = [
        {"run_label": "cinc_aug", "fragment": {"mcc": 0.80}, "patient": {"mcc": 0.82, "accuracy": 0.90}},
        {"run_label": "cinc_aug", "fragment": {"mcc": 0.84}, "patient": {"mcc": 0.86, "accuracy": 0.92}},
        {"run_label": "cinc_orig", "patient": {"mcc": 0.70, "accuracy": 0.85}},
    ]
    flat = flatten_metrics(records[0])
    assert flat["patient.mcc"] == 0.82 and flat["fragment.mcc"] == 0.80

    summary = summarize(records, group_by=["run_label"])
    mean, std, n = summary["run_label=cinc_aug"]["patient.mcc"]
    assert n == 2 and abs(mean - 0.84) < 1e-9 and std > 0
    table = to_markdown(summary, metrics=["mcc", "accuracy"])
    assert "condition" in table and "cinc_aug" in table
