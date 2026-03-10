"""Unit tests for the PySCF runner using mocked pyscf modules."""

# Standard Library
from types import SimpleNamespace

# Third Party Library
import numpy as np

# First Party Library
from wepy.runners.pyscf import PySCFRunner
from wepy.walker import Walker, WalkerState


class _FakeGradients(object):
    def __init__(self, gradients):
        self._gradients = gradients

    def kernel(self):
        return self._gradients


class _FakeMF(object):
    def __init__(self, gradients, energy):
        self._gradients = gradients
        self._energy = energy
        self.xc = None

    def kernel(self):
        return self._energy

    def nuc_grad_method(self):
        return _FakeGradients(self._gradients)


class _FakeModuleFactory(object):
    def __init__(self, gradients, energy):
        self._gradients = gradients
        self._energy = energy

    def gto(self):
        return SimpleNamespace(M=lambda **kwargs: kwargs)

    def scf(self):
        return SimpleNamespace(
            RHF=lambda mol: _FakeMF(self._gradients, self._energy),
            UHF=lambda mol: _FakeMF(self._gradients, self._energy),
        )

    def dft(self):
        return SimpleNamespace(
            RKS=lambda mol: _FakeMF(self._gradients, self._energy),
            UKS=lambda mol: _FakeMF(self._gradients, self._energy),
        )


def test_run_segment_updates_positions(monkeypatch):
    gradients = np.array([[1.0, -2.0, 0.5]])
    energy = -1.23
    fake = _FakeModuleFactory(gradients=gradients, energy=energy)

    def _fake_import(name):
        if name == "pyscf.gto":
            return fake.gto()
        if name == "pyscf.scf":
            return fake.scf()
        if name == "pyscf.dft":
            return fake.dft()
        raise ValueError(name)

    monkeypatch.setattr("wepy.runners.pyscf.importlib.import_module", _fake_import)

    runner = PySCFRunner(step_size=0.1)
    walker = Walker(
        WalkerState(symbols=["H"], positions=np.array([[0.0, 0.0, 0.0]])),
        0.5,
    )

    new_walker = runner.run_segment(walker, 2)

    np.testing.assert_allclose(
        new_walker.state["positions"],
        np.array([[-0.2, 0.4, -0.1]]),
    )
    assert new_walker.state["energy"] == energy
    np.testing.assert_allclose(new_walker.state["gradients"], gradients)
    assert new_walker.weight == walker.weight


def test_rks_requires_xc(monkeypatch):
    fake = _FakeModuleFactory(gradients=np.zeros((1, 3)), energy=0.0)

    def _fake_import(name):
        if name == "pyscf.gto":
            return fake.gto()
        if name == "pyscf.scf":
            return fake.scf()
        if name == "pyscf.dft":
            return fake.dft()
        raise ValueError(name)

    monkeypatch.setattr("wepy.runners.pyscf.importlib.import_module", _fake_import)

    runner = PySCFRunner(method="RKS")
    walker = Walker(
        WalkerState(symbols=["H"], positions=np.array([[0.0, 0.0, 0.0]])),
        1.0,
    )

    try:
        runner.run_segment(walker, 1)
    except ValueError as exc:
        assert "requires an xc functional" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing xc functional")
