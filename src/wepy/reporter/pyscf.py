"""Reporter helpers for PySCF based simulations."""

# Third Party Library
import numpy as np

# First Party Library
from wepy.reporter.dashboard import RunnerDashboardSection
from wepy.reporter.hdf5 import WepyHDF5Reporter
from wepy.runners.pyscf import UNIT_NAMES


class PySCFRunnerDashboardSection(RunnerDashboardSection):
    RUNNER_SECTION_TEMPLATE = """
Runner: {{ name }}

Backend: {{ backend }}
Dynamics mode: {{ dynamics_mode }}
Temperature (K): {{ temperature_kelvin }}
Step size: {{ step_size }}
Average Energy: {{ avg_energy }}
"""

    def __init__(self, runner=None, step_size=None, backend="cpu", dynamics_mode="steepest_descent", temperature_kelvin=None, **kwargs):
        if "name" not in kwargs:
            kwargs["name"] = "PySCFRunner"

        super().__init__(runner=runner, **kwargs)

        if runner is None:
            self.step_size = step_size
            self.backend = backend
            self.dynamics_mode = dynamics_mode
            self.temperature_kelvin = temperature_kelvin
        else:
            self.step_size = runner.step_size
            self.backend = runner.backend
            self.dynamics_mode = getattr(runner, "dynamics_mode", "steepest_descent")
            self.temperature_kelvin = getattr(runner, "temperature_kelvin", None)

        self._energies = []

    def update_values(self, **kwargs):
        energies = []
        for walker in kwargs.get("new_walkers", []):
            energy = walker.state.dict().get("energy", None)
            if energy is None:
                continue
            energies.append(float(np.asarray(energy).ravel()[0]))
        if len(energies) > 0:
            self._energies.extend(energies)

    def gen_fields(self, **kwargs):
        fields = super().gen_fields(**kwargs)

        avg_energy = np.nan
        if len(self._energies) > 0:
            avg_energy = float(np.mean(self._energies))

        fields.update(
            {
                "backend": self.backend,
                "step_size": self.step_size,
                "dynamics_mode": self.dynamics_mode,
                "temperature_kelvin": self.temperature_kelvin,
                "avg_energy": avg_energy,
            }
        )

        return fields


class PySCFHDF5Reporter(WepyHDF5Reporter):
    """HDF5 reporter preconfigured for PySCF walker state fields."""

    DEFAULT_SAVE_FIELDS = (
        "positions",
        "energy",
        "gradients",
        "density_matrix",
        "density_grid",
        "density_grid_origin",
        "density_grid_spacing",
        "segment_step_idx",
    )

    def __init__(
        self,
        save_fields=None,
        units=None,
        wepy_hdf5_path=None,
        file_paths=None,
        **kwargs,
    ):
        if save_fields is None:
            save_fields = self.DEFAULT_SAVE_FIELDS

        if units is None:
            units = dict(UNIT_NAMES)

        # Work around explicit-path handling in FileReporter by always
        # normalizing to file_paths for this single-file reporter.
        if file_paths is None and wepy_hdf5_path is not None:
            file_paths = [wepy_hdf5_path]

        super().__init__(
            save_fields=save_fields,
            units=units,
            file_paths=file_paths,
            **kwargs,
        )
