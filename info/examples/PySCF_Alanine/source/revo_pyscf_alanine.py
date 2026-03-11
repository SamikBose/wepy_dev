"""Set up a REVO simulation with PySCF dynamics for alanine dipeptide."""

# Standard Library
import argparse
import tempfile

# Third Party Library
import mdtraj as mdj
import numpy as np

# First Party Library
from wepy.boundary_conditions.boundary import NoBC
from wepy.reporter.dashboard import DashboardReporter
from wepy.reporter.pyscf import PySCFHDF5Reporter, PySCFRunnerDashboardSection
from wepy.resampling.distances.pyscf import QMGridDensityDistance
from wepy.resampling.resamplers.revo import REVOResampler
from wepy.runners.pyscf import PySCFRunner, PySCFState, PySCFWalker
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
    # mdtraj stores positions in nm, convert to angstrom
    positions = np.asarray(traj.xyz[0], dtype=float) * 10.0
    symbols = [atom.element.symbol for atom in topology.atoms]

    return topology, symbols, positions


def generate_initial_walkers(symbols, positions, n_walkers=5, jitter=0.01, seed=13):
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
            basis="sto-3g",
            method="RHF",
            unit="Angstrom",
            segment_step_idx=0,
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
        pmax=0.5,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-walkers", type=int, default=5)
    parser.add_argument("--n-cycles", type=int, default=2)
    parser.add_argument("--segment-length", type=int, default=1)
    parser.add_argument("--step-size", type=float, default=1e-4)
    parser.add_argument("--basis", type=str, default="sto-3g")
    parser.add_argument("--method", type=str, default="RHF")
    parser.add_argument("--xc", type=str, default=None)
    parser.add_argument("--backend", type=str, default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--disable-scanner", action="store_true")
    parser.add_argument("--h5-path", type=str, default="alanine_pyscf.wepy.h5")
    parser.add_argument("--dash-path", type=str, default="alanine_pyscf.dash.org")
    args = parser.parse_args()

    mdj_top, symbols, positions = parse_with_mdtraj_topology(ALANINE_DIPEPTIDE_PDB)
    walkers = generate_initial_walkers(symbols, positions, n_walkers=args.n_walkers)

    runner = PySCFRunner(
        basis=args.basis,
        method=args.method,
        xc=args.xc,
        step_size=args.step_size,
        backend=args.backend,
        use_scf_scanner=not args.disable_scanner,
    )

    resampler = build_revo_resampler(init_state=walkers[0].state)

    json_topology = mdtraj_to_json_topology(mdj_top)
    h5_reporter = PySCFHDF5Reporter(
        wepy_hdf5_path=args.h5_path,
        topology=json_topology,
        resampler=resampler,
        boundary_conditions=NoBC(),
    )

    dash_reporter = DashboardReporter(
        dashboard_path=args.dash_path,
        runner_dash=PySCFRunnerDashboardSection(runner=runner),
    )

    sim_manager = Manager(
        walkers,
        runner=runner,
        resampler=resampler,
        boundary_conditions=NoBC(),
        reporters=[h5_reporter, dash_reporter],
    )

    end_walkers, _ = sim_manager.run_simulation(
        n_cycles=args.n_cycles,
        segment_lengths=args.segment_length,
    )

    print(f"Completed REVO/PySCF run with {len(end_walkers)} walkers")
    print("Final walker energies:", [walker.state["energy"] for walker in end_walkers])


if __name__ == "__main__":
    main()
