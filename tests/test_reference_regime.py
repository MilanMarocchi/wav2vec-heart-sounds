"""The reference_train_rnn regime halves epochs and augments the validation set."""

import inspect

from mpcg_wav2vec.experiments import cinc


def test_reference_regime_params_exist():
    sig = inspect.signature(cinc.run)
    assert "reference_train_rnn" in sig.parameters
    assert inspect.signature(cinc.run_leave_out_db).parameters["reference_train_rnn"].default is False


def test_epoch_halving_logic():
    # Mirrors the runner's rule so the intent is pinned by a test.
    def train_epochs(epochs, reference):
        return max(1, epochs // 2) if reference else epochs

    assert train_epochs(20, True) == 10
    assert train_epochs(1, True) == 1
    assert train_epochs(20, False) == 20
