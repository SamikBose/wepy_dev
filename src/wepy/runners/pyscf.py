"""PySCF molecular simulation runner and accessory classes.

This module mirrors the architecture of :mod:`wepy.runners.openmm` with:

- :class:`PySCFRunner` implementing the Runner interface
- :class:`PySCFState` and :class:`PySCFWalker`
- CPU/GPU worker specializations for WorkerMapper and TaskMapper
"""

# Standard Library
import importlib
import logging
import os
from copy import deepcopy

# Third Party Library
import numpy as np

# First Party Library
from wepy.runners.runner import Runner
from wepy.walker import Walker, WalkerState
from wepy.work_mapper.task_mapper import WalkerTaskProcess
from wepy.work_mapper.worker import Worker

logger = logging.getLogger(__name__)

KEYS = (
    "symbols",
    "positions",
    "energy",
    "gradients",
    "charge",
    "spin",
    "basis",
    "method",
    "xc",
    "unit",
    "segment_step_idx",
)
"""Names of canonical fields in :class:`PySCFState`."""

UNIT_NAMES = (
    ("positions_unit", "angstrom"),
    ("energy_unit", "hartree"),
    ("gradients_unit", "hartree/bohr"),
)
"""Serialized unit names for PySCF state fields."""


class PySCFState(WalkerState):
    """WalkerState implementation for PySCF based simulations."""

    KEYS = KEYS

    def __init__(self, **kwargs):
        self._data = kwargs

    def __getitem__(self, key):
        return self._data[key]

    def dict(self):
        return deepcopy(self._data)


class PySCFWalker(Walker):
    """Walker enforcing use of :class:`PySCFState`."""

    def __init__(self, state, weight):
        assert isinstance(
            state, PySCFState
        ), "state must be an instance of PySCFState not {}".format(type(state))
        super().__init__(state, weight)


class PySCFRunner(Runner):
    """Runner that propagates coordinates using PySCF energies/gradients."""

    SUPPORTED_METHODS = ("RHF", "UHF", "RKS", "UKS")

    def __init__(
        self,
        basis="sto-3g",
        method="RHF",
        xc=None,
        charge=0,
        spin=0,
        unit="Angstrom",
        step_size=1e-3,
        backend="cpu",
        use_scf_scanner=True,
    ):
        self.basis = basis
        self.method = method.upper()
        self.xc = xc
        self.charge = charge
        self.spin = spin
        self.unit = unit
        self.step_size = step_size
        self.backend = backend
        self.use_scf_scanner = use_scf_scanner

        if self.method not in self.SUPPORTED_METHODS:
            raise ValueError(
                "Unsupported PySCF mean-field method "
                f"'{self.method}'. Must be one of: {self.SUPPORTED_METHODS}"
            )

        self._cycle_backend = None
        self._cycle_platform_kwargs = None
        self._last_cycle_segments_split_times = []

    def pre_cycle(self, backend=None, platform_kwargs=None, **kwargs):
        self._cycle_backend = backend
        self._cycle_platform_kwargs = platform_kwargs

    def post_cycle(self, **kwargs):
        self._cycle_backend = None
        self._cycle_platform_kwargs = None

    def _build_molecule(self, state):
        pyscf_gto = importlib.import_module("pyscf.gto")
        state_data = state.dict()

        symbols = state_data["symbols"]
        positions = np.asarray(state_data["positions"], dtype=float)
        atom = [(symbol, tuple(coord)) for symbol, coord in zip(symbols, positions)]

        return pyscf_gto.M(
            atom=atom,
            basis=state_data.get("basis", self.basis),
            charge=state_data.get("charge", self.charge),
            spin=state_data.get("spin", self.spin),
            unit=state_data.get("unit", self.unit),
        )

    def _build_mean_field(self, mol, state):
        state_data = state.dict()
        method = state_data.get("method", self.method).upper()

        pyscf_scf = importlib.import_module("pyscf.scf")
        pyscf_dft = importlib.import_module("pyscf.dft")

        if method == "RHF":
            mf = pyscf_scf.RHF(mol)
        elif method == "UHF":
            mf = pyscf_scf.UHF(mol)
        elif method == "RKS":
            mf = pyscf_dft.RKS(mol)
            xc = state_data.get("xc", self.xc)
            if xc is None:
                raise ValueError("RKS method requires an xc functional.")
            mf.xc = xc
        elif method == "UKS":
            mf = pyscf_dft.UKS(mol)
            xc = state_data.get("xc", self.xc)
            if xc is None:
                raise ValueError("UKS method requires an xc functional.")
            mf.xc = xc
        else:
            raise ValueError(f"Unsupported PySCF mean-field method '{method}'.")

        return mf

    def _configure_hardware(self, mf, backend="cpu", platform_kwargs=None):
        platform_kwargs = platform_kwargs or {}

        num_threads = platform_kwargs.get("Threads")
        if num_threads is not None:
            os.environ["OMP_NUM_THREADS"] = str(num_threads)

        if backend and str(backend).lower() == "gpu":
            device_id = platform_kwargs.get("DeviceIndex")
            if device_id is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

            if hasattr(mf, "to_gpu"):
                mf = mf.to_gpu()
            else:
                raise RuntimeError(
                    "Requested GPU backend but PySCF mean-field object does not "
                    "support to_gpu()."
                )

        return mf

    def _build_gradient_scanner(self, mf):
        grad_method = mf.nuc_grad_method()
        if not hasattr(grad_method, "as_scanner"):
            return None

        return grad_method.as_scanner()

    def generate_state(self, state_data, positions, energy, gradients, segment_step_idx):
        return PySCFState(
            **{
                **state_data,
                "positions": positions,
                "energy": energy,
                "gradients": gradients,
                "segment_step_idx": segment_step_idx,
            }
        )

    def run_segment(self, walker, segment_length, **kwargs):
        state_data = walker.state.dict()
        positions = np.asarray(state_data["positions"], dtype=float).copy()

        total_steps = int(segment_length)
        if total_steps < 0:
            raise ValueError("segment_length must be >= 0")

        backend = kwargs.get("backend", self._cycle_backend or self.backend)
        platform_kwargs = kwargs.get(
            "platform_kwargs", self._cycle_platform_kwargs or {}
        )

        last_energy = state_data.get("energy", None)
        last_gradients = np.zeros_like(positions)
        segment_step_idx = 0

        scanner = None
        if total_steps > 0 and self.use_scf_scanner:
            init_state = PySCFState(
                **{**state_data, "positions": positions, "segment_step_idx": 0}
            )
            init_mol = self._build_molecule(init_state)
            init_mf = self._build_mean_field(init_mol, init_state)
            init_mf = self._configure_hardware(
                init_mf, backend=backend, platform_kwargs=platform_kwargs
            )
            scanner = self._build_gradient_scanner(init_mf)

        for step_idx in range(1, total_steps + 1):
            iter_state = PySCFState(
                **{**state_data, "positions": positions, "segment_step_idx": step_idx}
            )
            mol = self._build_molecule(iter_state)

            if scanner is None:
                mf = self._build_mean_field(mol, iter_state)
                mf = self._configure_hardware(
                    mf, backend=backend, platform_kwargs=platform_kwargs
                )
                energy = mf.kernel()
                gradients = np.asarray(mf.nuc_grad_method().kernel(), dtype=float)
            else:
                energy, gradients = scanner(mol)
                gradients = np.asarray(gradients, dtype=float)

            positions = positions - self.step_size * gradients
            last_energy = float(energy)
            last_gradients = gradients
            segment_step_idx = step_idx

        new_state = self.generate_state(
            state_data,
            positions=positions,
            energy=last_energy,
            gradients=last_gradients,
            segment_step_idx=segment_step_idx,
        )

        if isinstance(walker, PySCFWalker):
            return PySCFWalker(new_state, walker.weight)
        return Walker(new_state, walker.weight)


class PySCFCPUWorker(Worker):
    """Worker specialization for CPU PySCF execution."""

    NAME_TEMPLATE = "PySCFCPUWorker-{}"
    DEFAULT_NUM_THREADS = 1

    def __init__(self, *args, **kwargs):
        if "num_threads" not in kwargs:
            num_threads = self.DEFAULT_NUM_THREADS
        else:
            num_threads = kwargs.pop("num_threads")

        super().__init__(*args, num_threads=num_threads, **kwargs)

    def run_task(self, task):
        platform_options = {"Threads": str(self.attributes["num_threads"])}
        return task(backend="cpu", platform_kwargs=platform_options)


class PySCFGPUWorker(Worker):
    """Worker specialization for GPU PySCF execution."""

    NAME_TEMPLATE = "PySCFGPUWorker-{}"

    def run_task(self, task):
        device_id = self.mapper_attributes["device_ids"][self._worker_idx]
        platform_options = {"DeviceIndex": str(device_id)}
        return task(backend="gpu", platform_kwargs=platform_options)


class PySCFCPUWalkerTaskProcess(WalkerTaskProcess):
    """Task-process specialization for CPU PySCF execution."""

    NAME_TEMPLATE = "PySCF_CPU_Walker_Task-{}"

    def run_task(self, task):
        if "num_threads" in self.mapper_attributes:
            num_threads = self.mapper_attributes["num_threads"]
            platform_options = {"Threads": str(num_threads)}
        else:
            platform_options = {}

        return task(backend="cpu", platform_kwargs=platform_options)


class PySCFGPUWalkerTaskProcess(WalkerTaskProcess):
    """Task-process specialization for GPU PySCF execution."""

    NAME_TEMPLATE = "PySCF_GPU_Walker_Task-{}"

    def run_task(self, task):
        device_id = self.mapper_attributes["device_ids"][self._worker_idx]
        platform_options = {"DeviceIndex": str(device_id)}
        return task(backend="gpu", platform_kwargs=platform_options)
