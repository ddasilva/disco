import warnings

from astropy import units, constants
import h5py
from matplotlib import pyplot as plt
import numpy as np

from disco._dimensionalization import (
    undim_time,
    undim_magnetic_field,
    undim_space,
    undim_energy,
    undim_momentum,
    dim_time,
    dim_magnetic_moment,
    dim_space,
    dim_energy,
    dim_momentum,
)

# Units for when variables are stored on disk
TIME_UNITS = units.s
SPACE_UNITS = constants.R_earth
MOMENTUM_UNITS = units.keV * units.s / units.m
MAGFIELD_UNITS = units.nT
MAGNETIC_MOMENT_UNITS = units.MeV / units.nT
ENERGY_UNITS = units.eV
MASS_UNITS = units.kg
CHARGE_UNITS = units.C

# Maximum number of particles before a warning is issued
MAX_PARTICLES_BEFORE_WARNING = 1000


class ParticleHistory:
    """History of particle states.

    Arrays are in the shape (n_time_steps, n_particles).
    Mass and charge are scalars.

    See also:
    * `disco.TraceConfig(output_freq=...)`: Controlling between how many
      iterations between particle state is saved.
    """

    def __init__(self, t, x, y, z, ppar, M, B, W, h, mass, charge):
        self.t = t.to(TIME_UNITS)
        self.x = x.to(SPACE_UNITS)
        self.y = y.to(SPACE_UNITS)
        self.z = z.to(SPACE_UNITS)
        self.ppar = ppar.to(MOMENTUM_UNITS)
        self.M = M.to(MAGNETIC_MOMENT_UNITS)
        self.B = B.to(MAGFIELD_UNITS)
        self.W = W.to(ENERGY_UNITS)
        self.h = h.to(TIME_UNITS)
        self.mass = mass.to(MASS_UNITS)
        self.charge = charge.to(CHARGE_UNITS)

    def save(self, hdf_path):
        """Save particle history to an HDF5 file.

        Args
          hdf_path: Path to the HDF5 file where the history will be saved.
        See also:
          `ParticleHistory.load`: Load particle history from an HDF5 file.
        """
        with h5py.File(hdf_path, "w") as hdf:
            hdf["t"] = self.t.to_value(TIME_UNITS)
            hdf["x"] = self.x.to_value(SPACE_UNITS)
            hdf["y"] = self.y.to_value(SPACE_UNITS)
            hdf["z"] = self.z.to_value(SPACE_UNITS)
            hdf["ppar"] = self.ppar.to_value(MOMENTUM_UNITS)
            hdf["M"] = self.M.to_value(MAGNETIC_MOMENT_UNITS)
            hdf["B"] = self.B.to_value(MAGFIELD_UNITS)
            hdf["W"] = self.W.to_value(ENERGY_UNITS)
            hdf["h"] = self.h.to_value(TIME_UNITS)
            hdf["mass"] = self.mass.to_value(MASS_UNITS)
            hdf["charge"] = self.charge.to_value(CHARGE_UNITS)

            hdf["t"].attrs["UNITS"] = TIME_UNITS.to_string()
            hdf["x"].attrs["UNITS"] = SPACE_UNITS.to_string()
            hdf["y"].attrs["UNITS"] = SPACE_UNITS.to_string()
            hdf["z"].attrs["UNITS"] = SPACE_UNITS.to_string()
            hdf["ppar"].attrs["UNITS"] = MOMENTUM_UNITS.to_string()
            hdf["M"].attrs["UNITS"] = MAGNETIC_MOMENT_UNITS.to_string()
            hdf["B"].attrs["UNITS"] = MAGFIELD_UNITS.to_string()
            hdf["W"].attrs["UNITS"] = ENERGY_UNITS.to_string()
            hdf["h"].attrs["UNITS"] = TIME_UNITS.to_string()
            hdf["mass"].attrs["UNITS"] = MASS_UNITS.to_string()
            hdf["charge"].attrs["UNITS"] = CHARGE_UNITS.to_string()

    @classmethod
    def load(cls, hdf_path):
        """Load particle history from an HDF5 file.

        Args
          hdf_path: Path to the HDF5 file from which the history will be loaded.
        Returns
          ParticleHistory: An instance of ParticleHistory containing the loaded data.
        See also:
          `ParticleHistory.save`: Save particle history to an HDF5 file.
        """
        with h5py.File(hdf_path, "r") as hdf:
            t = hdf["t"][:] * TIME_UNITS
            x = hdf["x"][:] * SPACE_UNITS
            y = hdf["y"][:] * SPACE_UNITS
            z = hdf["z"][:] * SPACE_UNITS
            ppar = hdf["ppar"][:] * MOMENTUM_UNITS
            M = hdf["M"][:] * MAGNETIC_MOMENT_UNITS
            B = hdf["B"][:] * MAGFIELD_UNITS
            W = hdf["W"][:] * ENERGY_UNITS
            h = hdf["h"][:] * TIME_UNITS
            mass = hdf["mass"][()] * MASS_UNITS
            charge = hdf["charge"][()] * CHARGE_UNITS

        return cls(t, x, y, z, ppar, M, B, W, h, mass, charge)

    def _plot_trajectory(
        self,
        ax,
        x_vals,
        y_vals,
        inds,
        endpoints,
        sample,
        earth,
        grid,
        title,
        xlabel,
        ylabel,
    ):
        """Helper function to plot particle trajectory in a 2D plane."""
        # Create a new figure and axis if none is provided
        if ax is None:
            _, ax = plt.subplots()

        # Determine indices to plot
        if inds is None:
            inds = np.arange(x_vals.shape[1], dtype=int)
        elif not isinstance(inds, np.ndarray):
            inds = np.array([inds])
        if sample is not None:
            np.random.shuffle(inds)
            inds = inds[:sample]

        # Issue warning if too many particles are being plotted
        if inds.size > MAX_PARTICLES_BEFORE_WARNING:
            warnings.warn(
                "Plotting more than 1000 points may be slow. Consider downsampling with sample=500."
            )

        # Ensure indices are within bounds
        if inds.max() >= x_vals.shape[1]:
            raise IndexError(
                f"Index {inds.max()} is out of bounds for the number of particles {x_vals.shape[1]}."
            )

        # Plot the trajectories or endpoints
        if endpoints:
            ax.plot(x_vals[-1, inds], y_vals[-1, inds], marker=".")
        else:
            for i in inds:
                ax.plot(x_vals[:, i], y_vals[:, i])

        # Draw earth if set
        if earth:
            earth_circle = plt.Circle((0, 0), 1, color="k", zorder=100)
            ax.add_patch(earth_circle)

        # Setup axis labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")

        # Setup grid if set
        if grid:
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.axvline(0, color="black", linewidth=0.5)

        return ax

    def plot_xy(
        self,
        ax=None,
        inds=None,
        endpoints=False,
        sample=None,
        earth=True,
        grid=True,
        title="Particle Trajectory in XY Plane",
    ):
        """Plot the particle trajectory in the XY plane.

        Args:
          ax: Matplotlib axis to plot on. If None, a new figure and axis will be created.
          inds: Indices of the points to plot. If None, all points will be plotted.
          endpoints: If True, plot only the start and end points of the trajectory.
          sample: If specified, randomly sample this many particles to plot.
          earth: If True, draw a circle representing the Earth at the origin.
          grid: If True, add a grid to the plot.
          title: Title of the plot.
        Returns
          ax: The axis with the plotted trajectory.
        """
        return self._plot_trajectory(
            ax,
            self.x.value,
            self.y.value,
            inds,
            endpoints,
            sample,
            earth,
            grid,
            title,
            "X ($R_E$)",
            "Y ($R_E$)",
        )

    def plot_xz(
        self,
        ax=None,
        inds=None,
        endpoints=False,
        sample=None,
        earth=True,
        grid=True,
        title="Particle Trajectory in XZ Plane",
    ):
        """Plot the particle trajectory in the XZ plane.

        Args:
          ax: Matplotlib axis to plot on. If None, a new figure and axis will be created.
          inds: Indices of the points to plot. If None, all points will be plotted.
          endpoints: If True, plot only the start and end points of the trajectory.
          sample: If specified, randomly sample this many particles to plot.
          earth: If True, draw a circle representing the Earth at the origin.
          grid: If True, add a grid to the plot.
          title: Title of the plot.
        Returns
          ax: The axis with the plotted trajectory.
        """
        return self._plot_trajectory(
            ax,
            self.x.value,
            self.z.value,
            inds,
            endpoints,
            sample,
            earth,
            grid,
            title,
            "X ($R_E$)",
            "Z ($R_E$)",
        )

    def plot_yz(
        self,
        ax=None,
        inds=None,
        endpoints=False,
        sample=None,
        earth=True,
        grid=True,
        title="Particle Trajectory in YZ Plane",
    ):
        """Plot the particle trajectory in the YZ plane.

        Args:
          ax: Matplotlib axis to plot on. If None, a new figure and axis will be created.
          inds: Indices of the points to plot. If None, all points will be plotted.
          endpoints: If True, plot only the start and end points of the trajectory.
          sample: If specified, randomly sample this many particles to plot.
          earth: If True, draw a circle representing the Earth at the origin.
          grid: If True, add a grid to the plot.
          title: Title of the plot.
        Returns
          ax: The axis with the plotted trajectory.
        """
        return self._plot_trajectory(
            ax,
            self.y.value,
            self.z.value,
            inds,
            endpoints,
            sample,
            earth,
            grid,
            title,
            "Y ($R_E$)",
            "Z ($R_E$)",
        )
