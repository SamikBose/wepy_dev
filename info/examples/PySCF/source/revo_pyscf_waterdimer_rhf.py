"""CPU-only REVO/PySCF water dimer example (RHF preset).

This version reads all run parameters from `pyscf_input.py` and mirrors the
structure of the alanine CPU example.
"""

# Standard Library
import tempfile

# Third Party Library
import mdtraj as mdj
import numpy as np

# First Party Library
from pyscf_input import WATER_DIMER_RHF_CONFIG as CONFIG
from wepy.boundary_conditions.boundary import NoBC
from wepy.reporter.dashboard import DashboardReporter
from wepy.reporter.pyscf import PySCFHDF5Reporter, PySCFRunnerDashboardSection
from wepy.resampling.distances.pyscf import ProtonTransfer
from wepy.resampling.resamplers.revo import REVOResampler
from wepy.runners.pyscf import PySCFCPUTaskMapper, PySCFRunner, PySCFState, PySCFWalker
from wepy.sim_manager import Manager
from wepy.util.mdtraj import mdtraj_to_json_topology

WATER_DIMER_PDB = """\
ATOM      1  O   HOH A   1       0.0000   0.0000   0.0000
ATOM      2  H1  HOH A   1       0.9580   0.0000   0.0000
ATOM      3  H2  HOH A   1      -0.2395   0.9275   0.0000
ATOM      4  O   HOH B   2       2.9760   0.0000   0.0000
ATOM      5  H1  HOH B   2       2.6565   0.7574   0.4920
ATOM      6  H2  HOH B   2       2.6565  -0.7574   0.4920
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
            xc=CONFIG.xc,
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
    break_pair = (0, 1)  # O_A - H_A1
    make_pair = (1, 3)  # H_A1 - O_B
    distance = ProtonTransfer(break_pair=break_pair, make_pair=make_pair)

    return REVOResampler(
        distance=distance,
        init_state=init_state,
        weights=True,
        pmax=0.99,
        dist_exponent=4,
        merge_dist=0.1,
        char_dist=0.1,
        merge_alg="pairs",
    )


def main():
    mdj_top, symbols, positions = parse_with_mdtraj_topology(WATER_DIMER_PDB)

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

    end_walkers, _ = sim_manager.run_simulation(
        n_cycles=CONFIG.n_cycles,
        segment_lengths=CONFIG.segment_length,
    )

    print(f"Completed REVO/PySCF water dimer RHF run with {len(end_walkers)} walkers")
    print(f"CPU workers: {num_workers}")
    print(f"Threads per worker: {CONFIG.cpu_num_threads_per_worker}")
    print("Final walker energies:", [walker.state["energy"] for walker in end_walkers])


if __name__ == "__main__":
    main()
