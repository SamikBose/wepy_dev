"""Run REVO/PySCF dynamics for a hydrogen-bonded water dimer (RHF preset).

This example uses the proton-transfer reaction coordinate as the REVO distance
metric:

    xi = d(O_A-H_A1) - d(H_A1-O_B)
"""

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
from wepy.resampling.distances.pyscf import ProtonTransfer
from wepy.resampling.resamplers.revo import REVOResampler
from wepy.runners.pyscf import PySCFRunner, PySCFState, PySCFWalker
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
    # mdtraj stores positions in nm, convert to angstrom
    positions = np.asarray(traj.xyz[0], dtype=float) * 10.0
    symbols = [atom.element.symbol for atom in topology.atoms]

    return topology, symbols, positions


def generate_initial_walkers(symbols, positions, n_walkers=8, jitter=0.005, seed=13):
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
            basis="6-31g*",
            method="RHF",
            unit="Angstrom",
            segment_step_idx=np.array([0], dtype=int),
            energy=np.array([np.nan]),
            gradients=np.zeros_like(noisy_positions),
            density_matrix=np.zeros((len(symbols), len(symbols))),
            density_grid=np.zeros((10, 10, 10)),
            density_grid_origin=np.zeros(3),
            density_grid_spacing=np.ones(3),
        )
        walkers.append(PySCFWalker(state, weight))

    return walkers


def build_revo_resampler(walker_state):
    break_pair = (0, 1)  # O_A - H_A1
    make_pair = (1, 3)  # H_A1 - O_B
    distance = ProtonTransfer(break_pair=break_pair, make_pair=make_pair)

    return REVOResampler(
        distance=distance,
        init_state=walker_state,
        weights=True,
        pmax=0.99,
        dist_exponent=4,
        merge_dist=0.1,
        char_dist=0.1,
        merge_alg="pairs",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-walkers", type=int, default=8)
    parser.add_argument("--n-cycles", type=int, default=5)
    parser.add_argument("--segment-length", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=1e-4)
    parser.add_argument(
        "--dynamics-mode",
        type=str,
        default="steepest_descent",
        choices=["steepest_descent", "langevin"],
    )
    parser.add_argument("--temperature-kelvin", type=float, default=300.0)
    parser.add_argument("--basis", type=str, default="6-31g*")
    parser.add_argument("--method", type=str, default="RHF", choices=["RHF", "UHF", "RKS", "UKS", "MP2", "DFMP2", "CCSD"])
    parser.add_argument("--xc", type=str, default="m06")
    parser.add_argument("--disable-scanner", action="store_true")
    parser.add_argument("--h5-path", type=str, default="waterdimer_pyscf.wepy.h5")
    parser.add_argument("--dash-path", type=str, default="waterdimer_pyscf.dash.org")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    mdj_top, symbols, positions = parse_with_mdtraj_topology(WATER_DIMER_PDB)
    walkers = generate_initial_walkers(
        symbols,
        positions,
        n_walkers=args.n_walkers,
    )

    runner = PySCFRunner(
        basis=args.basis,
        method=args.method,
        xc=args.xc,
        step_size=args.step_size,
        dynamics_mode=args.dynamics_mode,
        temperature_kelvin=args.temperature_kelvin,
        backend="cpu",
        use_scf_scanner=not args.disable_scanner,
        density_grid_shape=(10, 10, 10),
    )

    resampler = build_revo_resampler(walker_state=walkers[0].state)

    json_topology = mdtraj_to_json_topology(mdj_top)
    output_mode = "w" if args.overwrite else "x"

    h5_reporter = PySCFHDF5Reporter(
        file_paths=[args.h5_path],
        modes=[output_mode],
        topology=json_topology,
        resampler=resampler,
        boundary_conditions=NoBC(),
    )

    dash_reporter = DashboardReporter(
        file_paths=[args.dash_path],
        modes=[output_mode],
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

    print(f"Completed REVO/PySCF water-dimer run with {len(end_walkers)} walkers")
    print("Final walker energies:", [walker.state["energy"] for walker in end_walkers])


if __name__ == "__main__":
    main()
