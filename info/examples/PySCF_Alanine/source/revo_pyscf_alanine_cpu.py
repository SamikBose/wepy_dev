"""CPU-only REVO/PySCF alanine dipeptide example.

This version uses a separate `pyscf_input.py` file for all PySCF/simulation
parameters and performs walker-level CPU parallelization with TaskMapper.
"""

# Standard Library
import tempfile
from time import perf_counter

# Third Party Library
import mdtraj as mdj
import numpy as np

# First Party Library
from pyscf_input import CONFIG

from wepy.boundary_conditions.boundary import NoBC
from wepy.reporter.dashboard import DashboardReporter
from wepy.reporter.pyscf import PySCFHDF5Reporter, PySCFRunnerDashboardSection
from wepy.resampling.distances.pyscf import QMGridDensityDistance
from wepy.resampling.resamplers.revo import REVOResampler
from wepy.runners.pyscf import PySCFCPUTaskMapper, PySCFRunner, PySCFState, PySCFWalker
from wepy.sim_manager import Manager
from wepy.util.mdtraj import mdtraj_to_json_topology

ALANINE_DIPEPTIDE_PDB = """\
ATOM      1 1HH3 ACE     1       2.000   1.000  -0.000
ATOM      2  CH3 ACE     1       2.000   2.090   0.000
ATOM      3 2HH3 ACE     1       1.486   2.454   0.890
ATOM      4 3HH3 ACE     1       1.486   2.454  -0.890
ATOM      5  C   ACE     1       3.427   2.641  -0.000
ATOM      6  O   ACE     1       4.391   1.877  -0.000
ATOM      7  N   ALA     2       3.555   3.970  -0.000
ATOM      8  H   ALA     2       2.733   4.556  -0.000
ATOM      9  CA  ALA     2       4.853   4.614  -0.000
ATOM     10  HA  ALA     2       5.408   4.316   0.890
ATOM     11  CB  ALA     2       5.661   4.221  -1.232
ATOM     12 1HB  ALA     2       5.123   4.521  -2.131
ATOM     13 2HB  ALA     2       6.630   4.719  -1.206
ATOM     14 3HB  ALA     2       5.809   3.141  -1.241
ATOM     15  C   ALA     2       4.713   6.129   0.000
ATOM     16  O   ALA     2       3.601   6.653   0.000
ATOM     17  N   NME     3       5.846   6.835   0.000
ATOM     18  H   NME     3       6.737   6.359  -0.000
ATOM     19  CH3 NME     3       5.846   8.284   0.000
ATOM     20 1HH3 NME     3       4.819   8.648   0.000
ATOM     21 2HH3 NME     3       6.360   8.648   0.890
ATOM     22 3HH3 NME     3       6.360   8.648  -0.890
END
"""


def parse_with_mdtraj_topology(pdb_text):
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=True) as tmp:
        tmp.write(pdb_text)
        tmp.flush()
        traj = mdj.load_pdb(tmp.name)

    topology = traj.topology
    positions = np.asarray(traj.xyz[0], dtype=float) * 10.0
    symbols = [atom.element.symbol for atom in topology.atoms]

    return topology, symbols, positions


def generate_initial_walkers(
    symbols,
    positions,
    n_walkers,
    density_grid_shape,
    jitter,
    seed,
):
    rng = np.random.default_rng(seed)
    walkers = []
    weight = 1.0 / n_walkers

    for _ in range(n_walkers):
        noisy_positions = positions + rng.normal(scale=jitter, size=positions.shape)
        state = PySCFState(
            symbols=symbols,
            positions=noisy_positions,
            charge=0,
            spin=0,
            basis=CONFIG.basis,
            method=CONFIG.method,
            unit="Angstrom",
            segment_step_idx=np.array([0], dtype=int),
            energy=np.array([np.nan]),
            gradients=np.zeros_like(noisy_positions),
            density_matrix=np.zeros((len(symbols), len(symbols))),
            density_grid=np.zeros(density_grid_shape),
            density_grid_origin=np.zeros(3),
            density_grid_spacing=np.ones(3),
        )
        walkers.append(PySCFWalker(state, weight))

    return walkers


def build_revo_resampler(init_state):
    distance = QMGridDensityDistance(grid_key="density_grid", normalize=True)

    return REVOResampler(
        distance=distance,
        init_state=init_state,
        merge_dist=0.5,
        char_dist=1.0,
        pmin=1e-12,
        pmax=0.99,
    )


def main():
    mdj_top, symbols, positions = parse_with_mdtraj_topology(ALANINE_DIPEPTIDE_PDB)

    walkers = generate_initial_walkers(
        symbols=symbols,
        positions=positions,
        n_walkers=CONFIG.n_walkers,
        density_grid_shape=CONFIG.density_grid_shape,
        jitter=CONFIG.jitter,
        seed=CONFIG.seed,
    )

    runner = PySCFRunner(
        basis=CONFIG.basis,
        method=CONFIG.method,
        xc=CONFIG.xc,
        step_size=CONFIG.step_size,
        dynamics_mode=CONFIG.dynamics_mode,
        temperature_kelvin=CONFIG.temperature_kelvin,
        backend="cpu",
        use_scf_scanner=CONFIG.use_scf_scanner,
        density_grid_shape=CONFIG.density_grid_shape,
    )

    resampler = build_revo_resampler(init_state=walkers[0].state)

    json_topology = mdtraj_to_json_topology(mdj_top)
    output_mode = "w" if CONFIG.overwrite else "x"

    h5_reporter = PySCFHDF5Reporter(
        file_paths=[CONFIG.h5_path],
        modes=[output_mode],
        topology=json_topology,
        resampler=resampler,
        boundary_conditions=NoBC(),
    )

    dash_reporter = DashboardReporter(
        file_paths=[CONFIG.dash_path],
        modes=[output_mode],
        runner_dash=PySCFRunnerDashboardSection(runner=runner),
    )

    num_workers = CONFIG.cpu_num_workers or CONFIG.n_walkers
    mapper = PySCFCPUTaskMapper(
        num_workers=num_workers,
        num_threads=CONFIG.cpu_num_threads_per_worker,
    )

    sim_manager = Manager(
        walkers,
        runner=runner,
        work_mapper=mapper,
        resampler=resampler,
        boundary_conditions=NoBC(),
        reporters=[h5_reporter, dash_reporter],
    )

    time = perf_counter()
    end_walkers, _ = sim_manager.run_simulation(
        n_cycles=CONFIG.n_cycles,
        segment_lengths=CONFIG.segment_length,
    )

    print(f"Completed REVO/PySCF CPU run with {len(end_walkers)} walkers in {perf_counter() - time:.3f} seconds")
    print(f"CPU workers: {num_workers}")
    print(f"Threads per worker: {CONFIG.cpu_num_threads_per_worker}")
    print("Final walker energies:", [walker.state["energy"] for walker in end_walkers])


if __name__ == "__main__":
    main()
