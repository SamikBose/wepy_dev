"""PySCF-based runner implementation.

This runner evaluates electronic structure energies/gradients with
`pyscf` and performs simple gradient-descent style coordinate updates.
"""

# Standard Library
import importlib

# Third Party Library
import numpy as np

# First Party Library
from wepy.runners.runner import Runner
from wepy.walker import Walker, WalkerState


class PySCFRunner(Runner):
    """Runner for propagating walkers with PySCF calculations.

    The input walker state must include at least:

    - ``symbols``: iterable of atomic symbols (e.g. ["H", "H"])
    - ``positions``: shape (n_atoms, 3) coordinates

    Optional walker-state fields that override runner defaults:

    - ``charge``
    - ``spin``
    - ``basis``
    - ``method`` (RHF, UHF, RKS, UKS)
    - ``xc`` (required for RKS/UKS)
    - ``unit`` (default ``Angstrom``)
    """

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
    ):
        self.basis = basis
        self.method = method.upper()
        self.xc = xc
        self.charge = charge
        self.spin = spin
        self.unit = unit
        self.step_size = step_size

        if self.method not in self.SUPPORTED_METHODS:
            raise ValueError(
                "Unsupported PySCF mean-field method "
                f"'{self.method}'. Must be one of: {self.SUPPORTED_METHODS}"
            )

    def _build_molecule(self, state):
        pyscf_gto = importlib.import_module("pyscf.gto")

        symbols = state["symbols"]
        positions = np.asarray(state["positions"], dtype=float)

        atom = [(symbol, tuple(coord)) for symbol, coord in zip(symbols, positions)]

        return pyscf_gto.M(
            atom=atom,
            basis=state.dict().get("basis", self.basis),
            charge=state.dict().get("charge", self.charge),
            spin=state.dict().get("spin", self.spin),
            unit=state.dict().get("unit", self.unit),
        )

    def _build_mean_field(self, mol, state):
        method = state.dict().get("method", self.method).upper()

        pyscf_scf = importlib.import_module("pyscf.scf")
        pyscf_dft = importlib.import_module("pyscf.dft")

        if method == "RHF":
            mf = pyscf_scf.RHF(mol)
        elif method == "UHF":
            mf = pyscf_scf.UHF(mol)
        elif method == "RKS":
            mf = pyscf_dft.RKS(mol)
            xc = state.dict().get("xc", self.xc)
            if xc is None:
                raise ValueError("RKS method requires an xc functional.")
            mf.xc = xc
        elif method == "UKS":
            mf = pyscf_dft.UKS(mol)
            xc = state.dict().get("xc", self.xc)
            if xc is None:
                raise ValueError("UKS method requires an xc functional.")
            mf.xc = xc
        else:
            raise ValueError(f"Unsupported PySCF mean-field method '{method}'.")

        return mf

    def run_segment(self, walker, segment_length, **kwargs):
        """Run repeated PySCF evaluations and update coordinates.

        Each step runs an SCF calculation, evaluates nuclear gradients,
        and applies a steepest-descent coordinate update.
        """

        state_data = walker.state.dict()
        positions = np.asarray(state_data["positions"], dtype=float).copy()

        total_steps = int(segment_length)
        if total_steps < 0:
            raise ValueError("segment_length must be >= 0")

        last_energy = state_data.get("energy", None)
        last_gradients = np.zeros_like(positions)

        for _ in range(total_steps):
            iter_state = WalkerState(**{**state_data, "positions": positions})
            mol = self._build_molecule(iter_state)
            mf = self._build_mean_field(mol, iter_state)

            energy = mf.kernel()
            gradients = np.asarray(mf.nuc_grad_method().kernel(), dtype=float)

            positions = positions - self.step_size * gradients
            last_energy = float(energy)
            last_gradients = gradients

        new_state = WalkerState(
            **{
                **state_data,
                "positions": positions,
                "energy": last_energy,
                "gradients": last_gradients,
            }
        )

        return Walker(new_state, walker.weight)
