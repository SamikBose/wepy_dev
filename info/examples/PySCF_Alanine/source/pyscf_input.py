"""Input configuration for CPU-only REVO/PySCF alanine runs.

Edit this file instead of passing command-line arguments.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class PySCFInput:
    # Simulation size
    n_walkers: int = 5
    n_cycles: int = 2
    segment_length: int = 1

    # Walker initialization
    jitter: float = 0.01
    seed: int = 13

    # PySCF runner parameters
    basis: str = "sto-3g"
    method: str = "RHF"
    xc: Optional[str] = None
    step_size: float = 1e-4
    # 'steepest_descent' performs deterministic energy minimization.
    # 'langevin' adds thermal noise to approximate finite-temperature sampling.
    dynamics_mode: str = "steepest_descent"
    temperature_kelvin: float = 300.0
    use_scf_scanner: bool = True
    density_grid_shape: Tuple[int, int, int] = (10, 10, 10)

    # CPU walker-level parallelization
    # If None, defaults to n_walkers (i.e., one worker per walker when possible).
    cpu_num_workers: Optional[int] = None
    # Number of BLAS/OpenMP threads per walker process.
    # Set to 1 for maximal walker-level concurrency.
    cpu_num_threads_per_worker: int = 1

    # Output control
    h5_path: str = "alanine_pyscf_cpu.wepy.h5"
    dash_path: str = "alanine_pyscf_cpu.dash.org"
    overwrite: bool = True


CONFIG = PySCFInput()
