"""PySCF molecular simulation runner and accessory classes."""

# Standard Library
import importlib
import importlib.util
import logging
import os
from copy import deepcopy

# Third Party Library
import numpy as np

# First Party Library
from wepy.runners.runner import Runner
from wepy.walker import Walker, WalkerState
from wepy.work_mapper.task_mapper import TaskMapper, WalkerTaskProcess
from wepy.work_mapper.worker import Worker

logger = logging.getLogger(__name__)

KEYS = (
    "symbols",
    "positions",
    "energy",
    "gradients",
    "density_matrix",
    "density_grid",
    "density_grid_origin",
    "density_grid_spacing",
    "charge",
    "spin",
    "basis",
    "method",
    "xc",
    "unit",
    "segment_step_idx",
)

UNIT_NAMES = (
    ("positions_unit", "angstrom"),
    ("energy_unit", "hartree"),
    ("gradients_unit", "hartree/bohr"),
    ("density_grid_unit", "electron/bohr^3"),
)


class PySCFState(WalkerState):
    KEYS = KEYS

    def __init__(self, **kwargs):
        self._data = kwargs

    def __getitem__(self, key):
        return self._data[key]

    def dict(self):
        return deepcopy(self._data)


class PySCFWalker(Walker):
    def __init__(self, state, weight):
        assert isinstance(
            state, PySCFState
        ), "state must be an instance of PySCFState not {}".format(type(state))
        super().__init__(state, weight)


class PySCFRunner(Runner):
    SUPPORTED_METHODS = ("RHF", "UHF", "RKS", "UKS")
    SUPPORTED_DYNAMICS_MODES = ("steepest_descent", "langevin")
    BOLTZMANN_HARTREE_PER_K = 3.166811563e-6

    def __init__(
        self,
        basis="sto-3g",
        method="RHF",
        xc=None,
        charge=0,
        spin=0,
        unit="Angstrom",
        step_size=1e-3,
        dynamics_mode="steepest_descent",
        temperature_kelvin=300.0,
        random_seed=None,
        backend="cpu",
        use_scf_scanner=True,
        density_grid_shape=(10, 10, 10),
        density_grid_padding=2.0,
        gpu_fallback_cpu_on_error=False,
    ):
        self.basis = basis
        self.method = method.upper()
        self.xc = xc
        self.charge = charge
        self.spin = spin
        self.unit = unit
        self.step_size = step_size
        self.dynamics_mode = str(dynamics_mode).lower()
        self.temperature_kelvin = float(temperature_kelvin)
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.backend = backend
        self.use_scf_scanner = use_scf_scanner
        self.density_grid_shape = tuple(density_grid_shape)
        self.density_grid_padding = float(density_grid_padding)
        self.gpu_fallback_cpu_on_error = gpu_fallback_cpu_on_error

        if self.method not in self.SUPPORTED_METHODS:
            raise ValueError(
                "Unsupported PySCF mean-field method "
                f"'{self.method}'. Must be one of: {self.SUPPORTED_METHODS}"
            )

        if self.dynamics_mode not in self.SUPPORTED_DYNAMICS_MODES:
            raise ValueError(
                "Unsupported PySCF dynamics mode "
                f"'{self.dynamics_mode}'. Must be one of: {self.SUPPORTED_DYNAMICS_MODES}"
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
                try:
                    mf = mf.to_gpu()
                except ModuleNotFoundError as exc:
                    if getattr(exc, "name", None) == "cupy":
                        raise RuntimeError(
                            "GPU backend requested but CuPy is not installed. "
                            "Install a CuPy build compatible with your CUDA version "
                            "(e.g. cupy-cuda12x) or run with CPU backend."
                        ) from exc
                    raise
            else:
                raise RuntimeError(
                    "Requested GPU backend but PySCF mean-field object does not support to_gpu()."
                )

        return mf

    def _is_gpu_runtime_error(self, exc):
        msg = str(exc).lower()
        gpu_signatures = (
            "unsupported toolchain",
            "failed in block_diag kernel",
            "cuda error",
        )
        return any(sig in msg for sig in gpu_signatures)

    def _build_gradient_scanner(self, mf):
        grad_method = mf.nuc_grad_method()
        if not hasattr(grad_method, "as_scanner"):
            return None

        return grad_method.as_scanner()

    def _make_density_grid_coords(self, positions):
        mins = np.min(positions, axis=0) - self.density_grid_padding
        maxs = np.max(positions, axis=0) + self.density_grid_padding

        axes = [
            np.linspace(mins[i], maxs[i], self.density_grid_shape[i]) for i in range(3)
        ]
        mesh = np.meshgrid(*axes, indexing="ij")
        coords = np.stack(mesh, axis=-1).reshape(-1, 3)

        spacing = np.array(
            [axes[i][1] - axes[i][0] if len(axes[i]) > 1 else 1.0 for i in range(3)]
        )

        return coords, mins, spacing

    def _compute_density_grid(self, mol, density_matrix, positions):
        numint = importlib.import_module("pyscf.dft.numint")

        dm = np.asarray(density_matrix)
        if dm.ndim == 3:
            dm = dm[0] + dm[1]

        grid_coords, origin, spacing = self._make_density_grid_coords(positions)

        ao_values = numint.eval_ao(mol, grid_coords)
        rho = numint.eval_rho(mol, ao_values, dm)
        rho_grid = np.asarray(rho, dtype=float).reshape(self.density_grid_shape)

        return rho_grid, origin, spacing

    def generate_state(
        self,
        state_data,
        positions,
        energy,
        gradients,
        segment_step_idx,
        density_matrix,
        density_grid,
        density_grid_origin,
        density_grid_spacing,
    ):
        return PySCFState(
            **{
                **state_data,
                "positions": positions,
                "energy": np.array([np.nan if energy is None else float(np.asarray(energy).ravel()[0])]),
                "gradients": gradients,
                "segment_step_idx": np.array([int(segment_step_idx)]),
                "density_matrix": density_matrix,
                "density_grid": density_grid,
                "density_grid_origin": density_grid_origin,
                "density_grid_spacing": density_grid_spacing,
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
        last_density_matrix = np.zeros((positions.shape[0], positions.shape[0]))
        last_density_grid = np.zeros(self.density_grid_shape)
        last_density_grid_origin = np.zeros(3)
        last_density_grid_spacing = np.ones(3)
        segment_step_idx = 0

        scanner = None
        allow_gpu_fallback = kwargs.get(
            "gpu_fallback_cpu_on_error", self.gpu_fallback_cpu_on_error
        )

        if total_steps > 0 and self.use_scf_scanner:
            init_state = PySCFState(
                **{**state_data, "positions": positions, "segment_step_idx": 0}
            )
            init_mol = self._build_molecule(init_state)
            init_mf = self._build_mean_field(init_mol, init_state)
            try:
                init_mf = self._configure_hardware(
                    init_mf, backend=backend, platform_kwargs=platform_kwargs
                )
                scanner = self._build_gradient_scanner(init_mf)
            except RuntimeError as exc:
                if backend == "gpu" and allow_gpu_fallback and self._is_gpu_runtime_error(exc):
                    logger.warning(
                        "GPU initialization failed (%s); falling back to CPU for this segment.",
                        exc,
                    )
                    backend = "cpu"
                    platform_kwargs = {}
                    scanner = None
                else:
                    raise

        for step_idx in range(1, total_steps + 1):
            iter_state = PySCFState(
                **{**state_data, "positions": positions, "segment_step_idx": step_idx}
            )
            mol = self._build_molecule(iter_state)

            try:
                if scanner is None:
                    mf = self._build_mean_field(mol, iter_state)
                    mf = self._configure_hardware(
                        mf, backend=backend, platform_kwargs=platform_kwargs
                    )
                    energy = mf.kernel()
                    gradients = np.asarray(mf.nuc_grad_method().kernel(), dtype=float)
                    density_matrix = np.asarray(mf.make_rdm1(), dtype=float)
                else:
                    energy, gradients = scanner(mol)
                    gradients = np.asarray(gradients, dtype=float)

                    scan_base = getattr(scanner, "base", None)
                    if scan_base is None or not hasattr(scan_base, "make_rdm1"):
                        # conservative fallback if scanner wrapper does not expose base
                        mf = self._build_mean_field(mol, iter_state)
                        mf = self._configure_hardware(
                            mf, backend=backend, platform_kwargs=platform_kwargs
                        )
                        mf.kernel()
                        density_matrix = np.asarray(mf.make_rdm1(), dtype=float)
                    else:
                        density_matrix = np.asarray(scan_base.make_rdm1(), dtype=float)
            except RuntimeError as exc:
                if backend == "gpu" and allow_gpu_fallback and self._is_gpu_runtime_error(exc):
                    logger.warning(
                        "GPU execution failed (%s); retrying this step on CPU.",
                        exc,
                    )
                    backend = "cpu"
                    platform_kwargs = {}
                    scanner = None

                    mf = self._build_mean_field(mol, iter_state)
                    mf = self._configure_hardware(
                        mf, backend=backend, platform_kwargs=platform_kwargs
                    )
                    energy = mf.kernel()
                    gradients = np.asarray(mf.nuc_grad_method().kernel(), dtype=float)
                    density_matrix = np.asarray(mf.make_rdm1(), dtype=float)
                else:
                    raise

            density_grid, density_origin, density_spacing = self._compute_density_grid(
                mol, density_matrix, positions
            )

            positions = self._propagate_positions(positions, gradients)
            last_energy = float(energy)
            last_gradients = gradients
            last_density_matrix = density_matrix
            last_density_grid = density_grid
            last_density_grid_origin = density_origin
            last_density_grid_spacing = density_spacing
            segment_step_idx = step_idx

        new_state = self.generate_state(
            state_data,
            positions=positions,
            energy=last_energy,
            gradients=last_gradients,
            segment_step_idx=segment_step_idx,
            density_matrix=last_density_matrix,
            density_grid=last_density_grid,
            density_grid_origin=last_density_grid_origin,
            density_grid_spacing=last_density_grid_spacing,
        )

        if isinstance(walker, PySCFWalker):
            return PySCFWalker(new_state, walker.weight)
        return Walker(new_state, walker.weight)

    def _propagate_positions(self, positions, gradients):
        """Update coordinates using either steepest descent or overdamped Langevin updates.

        Notes
        -----
        ``steepest_descent`` is a deterministic geometry-relaxation update and does not
        sample a finite-temperature ensemble.
        ``langevin`` adds a stochastic term so trajectories include thermal fluctuations.
        """

        drift = self.step_size * gradients
        if self.dynamics_mode == "steepest_descent":
            return positions - drift

        noise_scale = np.sqrt(
            2.0 * self.BOLTZMANN_HARTREE_PER_K * self.temperature_kelvin * self.step_size
        )
        thermal_noise = self.rng.normal(0.0, noise_scale, size=positions.shape)
        return positions - drift + thermal_noise


class PySCFCPUWorker(Worker):
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
    NAME_TEMPLATE = "PySCFGPUWorker-{}"

    def run_task(self, task):
        device_id = self.mapper_attributes["device_ids"][self._worker_idx]
        platform_options = {"DeviceIndex": str(device_id)}
        return task(backend="gpu", platform_kwargs=platform_options)


class PySCFCPUWalkerTaskProcess(WalkerTaskProcess):
    NAME_TEMPLATE = "PySCF_CPU_Walker_Task-{}"

    def run_task(self, task):
        if "num_threads" in self.mapper_attributes:
            num_threads = self.mapper_attributes["num_threads"]
            platform_options = {"Threads": str(num_threads)}
        else:
            platform_options = {}

        return task(backend="cpu", platform_kwargs=platform_options)


class PySCFGPUWalkerTaskProcess(WalkerTaskProcess):
    NAME_TEMPLATE = "PySCF_GPU_Walker_Task-{}"

    def run_task(self, task):
        device_id = self.mapper_attributes["device_ids"][self._worker_idx]
        platform_options = {"DeviceIndex": str(device_id)}
        return task(backend="gpu", platform_kwargs=platform_options)


class PySCFCPUTaskMapper(TaskMapper):
    """Convenience TaskMapper for CPU walker-level parallelism."""

    def __init__(self, num_workers=None, num_threads=1, **kwargs):
        super().__init__(
            walker_task_type=PySCFCPUWalkerTaskProcess,
            num_workers=num_workers,
            num_threads=num_threads,
            **kwargs,
        )


class PySCFGPUTaskMapper(TaskMapper):
    """Convenience TaskMapper for GPU walker-level parallelism."""

    def __init__(self, num_workers=None, platform="CUDA", device_ids=None, **kwargs):
        if device_ids is None:
            raise ValueError("device_ids must be provided for PySCFGPUTaskMapper")

        if num_workers is None:
            num_workers = len(device_ids)

        super().__init__(
            walker_task_type=PySCFGPUWalkerTaskProcess,
            num_workers=num_workers,
            platform=platform,
            device_ids=device_ids,
            **kwargs,
        )
