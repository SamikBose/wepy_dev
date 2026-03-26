"""Input configuration for CPU-only REVO/PySCF examples.

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
    basis: str = "6-31g*"
    method: str = "RHF"
    # Allowed methods include RHF/UHF, RKS/UKS, MP2/DFMP2, and CCSD.
    xc: Optional[str] = "m06"
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


@dataclass
class WaterDimerInput(PySCFInput):
    n_walkers: int = 8
    n_cycles: int = 5
    segment_length: int = 2
    jitter: float = 0.005
    method: str = "RHF"
    xc: Optional[str] = "m06"
    h5_path: str = "waterdimer_pyscf.wepy.h5"
    dash_path: str = "waterdimer_pyscf.dash.org"


CONFIG = PySCFInput()
WATER_DIMER_RHF_CONFIG = WaterDimerInput(method="RHF", h5_path="waterdimer_rhf_pyscf.wepy.h5", dash_path="waterdimer_rhf_pyscf.dash.org")
WATER_DIMER_RKS_M06_CONFIG = WaterDimerInput(method="RKS", xc="m06", h5_path="waterdimer_rks_m06_pyscf.wepy.h5", dash_path="waterdimer_rks_m06_pyscf.dash.org")
