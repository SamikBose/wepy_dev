"""Input configuration for CPU-only REVO/PySCF alanine runs.

Edit this file instead of passing command-line arguments.
"""

import os
from dataclasses import dataclass


@dataclass
class PySCFInput:
    # Simulation size
    n_walkers: int = 5
    n_cycles: int = 10
    segment_length: int = 1

    # Walker initialization
    jitter: float = 0.01
    seed: int = 13

    # PySCF runner parameters
    basis: str = "sto-3g"
    method: str = "RHF"
    xc: str | None = None
    step_size: float = 1e-4
    # 'steepest_descent' performs deterministic energy minimization.
    # 'langevin' adds thermal noise to approximate finite-temperature sampling.
    #dynamics_mode: str = "steepest_descent"
    dynamics_mode: str = "langevin"
    temperature_kelvin: float = 300.0
    use_scf_scanner: bool = True
    density_grid_shape: tuple[int, int, int] = (10, 10, 10)

    # CPU walker-level parallelization
    # If None, defaults to n_walkers (i.e., one worker per walker when possible).
    cpu_num_workers: int | None = None

    _omp_threads_env_var = os.environ.get("OMP_NUM_THREADS")

    # Output control
    write_h5: bool = True
    write_dash: bool = True
    h5_path: str = f"alanine_cpu_{n_walkers}W_{n_cycles}C_{dynamics_mode}_{_omp_threads_env_var}T.wepy.h5"
    dash_path: str = f"alanine_cpu_{n_walkers}W_{n_cycles}C_{dynamics_mode}_{_omp_threads_env_var}T.dash.org"
    overwrite: bool = True


CONFIG = PySCFInput()
