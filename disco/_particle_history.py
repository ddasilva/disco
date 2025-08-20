import warnings

from astropy import units, constants
import cupy as cp
import h5py
from matplotlib import pyplot as plt
import numpy as np

from disco._dimensionalization import (
    undim_time,
    undim_magnetic_field,
    undim_space,
    undim_energy,
    undim_momentum,
    undim_magnetic_moment,
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
    """History of trajectory tracing.

    Arrays are in the shape (n_time_steps, n_particles).
    Mass and charge are scalars.

    If some parameters completed integration in fewer timesteps than others (such as
    by reaching the integration limit or going out of bounds), the last item will be
    duplicated to match the shape of the other arrays. The last item before duplication
    can be found by checking the `stopped` array, which is a boolean array of the same
    shape as the other arrays.

    Output can be stored and read from disk in HDF5 format. See the `load()` and
    `save()` methods.

    Attributes
    ----------
    t : array with units
      Time of the particle state at each step.
    x : array with units
      X position of the particle at each step.
    y : array with units
      Y position of the particle at each step.
    z : array with units
      Z position of the particle at each step.
    ppar : array with units
      Parallel momentum of the particle at each step.
    M : array with units
      Magnetic moment of the particle at each step.
    B : array with units
      Magnetic field at the particle position at each step.
    W : array with units
      Total energy of the particle at each step.
    h : array with units
      Adapative step size used in the integration at each step.
    stopped : array of bool
      Boolean array indicating whether the particle stopped at each step.
    extra_fields : dict
      Dictionary of additional fields computed during the trajectory tracing.
      Keys are field names and values are arrays with the same shape as the other arrays.
    mass : scalar with units
      Mass of the particles (constant).
    charge : scalar with units
      Charge of the particles (constant).

    Notes
    -----
    See `disco.TraceConfig(output_freq=...)`: for controlling between how many
    iterations between particle state is saved. If `output_freq` is set to `None`
    (the default), only the first and last points of the trace will be saved.

    Examples
    --------
    Saving output to disk:

    >>> hist = disco.trace_trajectory(config, particle_state, field_model)
    >>> hist.save("particle_history.h5")

    Loading output from disk and plotting:

    >>> hist = disco.ParticleHistory.load("particle_history.h5")
    >>> hist.plot_xz()
    >>> plt.savefig('myplot.png')
    """

    def __init__(self, t, x, y, z, ppar, M, B, W, h, stopped, mass, charge, extra_fields=None):
        self.t = t.to(TIME_UNITS)
        self.x = x.to(SPACE_UNITS)
        self.y = y.to(SPACE_UNITS)
        self.z = z.to(SPACE_UNITS)
        self.ppar = ppar.to(MOMENTUM_UNITS)
        self.M = M.to(MAGNETIC_MOMENT_UNITS)
        self.B = B.to(MAGFIELD_UNITS)
        self.W = W.to(ENERGY_UNITS)
        self.h = h.to(TIME_UNITS)
        self.stopped = stopped.astype(bool)

        # Ensure mass and charge are scalars with units
        self.mass = mass.to(MASS_UNITS)
        self.charge = charge.to(CHARGE_UNITS)

        # If extra_fields is None, initialize as an empty dictionary
        if extra_fields is None:
            self.extra_fields = {}
        else:
            self.extra_fields = extra_fields

    def save(self, hdf_path):
        """Save particle history to an HDF5 file.

        Parameters
        ----------
        hdf_path: str
          Path to the HDF5 file where the history will be saved.

        Notes
        -----
        See `ParticleHistory.load()`: to load particle history from an HDF5 file.
        """
        with h5py.File(hdf_path, "w") as hdf:
            # Set main fields
            hdf["t"] = self.t.to_value(TIME_UNITS)
            hdf["x"] = self.x.to_value(SPACE_UNITS)
            hdf["y"] = self.y.to_value(SPACE_UNITS)
            hdf["z"] = self.z.to_value(SPACE_UNITS)
            hdf["ppar"] = self.ppar.to_value(MOMENTUM_UNITS)
            hdf["M"] = self.M.to_value(MAGNETIC_MOMENT_UNITS)
            hdf["B"] = self.B.to_value(MAGFIELD_UNITS)
            hdf["W"] = self.W.to_value(ENERGY_UNITS)
            hdf["h"] = self.h.to_value(TIME_UNITS)
            hdf["stopped"] = self.stopped
            hdf["mass"] = self.mass.to_value(MASS_UNITS)
            hdf["charge"] = self.charge.to_value(CHARGE_UNITS)

            # Set extra fields if they exist
            if self.extra_fields:
                extra_fields_group = hdf.create_group("extra_fields")
                for key, value in self.extra_fields.items():
                    extra_fields_group[key] = value

            # Set attributes for units
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

        Parameters
        ----------
        hdf_path: str
          Path to the HDF5 file from which the history will be loaded.

        Returns
        -------
         An instance of `ParticleHistory` containing the loaded data.

        Notes
        -----
        See `ParticleHistory.save()` to save particle history to an HDF5 file.
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
            stopped = hdf["stopped"][:].astype(bool)
            mass = hdf["mass"][()] * MASS_UNITS
            charge = hdf["charge"][()] * CHARGE_UNITS

            if "extra_fields" in hdf.keys():
                extra_fields = {}

                for key, value in hdf["extra_fields"].items():
                    extra_fields[key] = value[:]
            else:
                extra_fields = None

        return cls(t, x, y, z, ppar, M, B, W, h, stopped, mass, charge, extra_fields=extra_fields)

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

        Parameters
        ----------
        ax: matplotlib axes
          Matplotlib axis to plot on. If None, a new figure and axis will be created.
        inds: int or list of ints
          Indices of the points to plot. If None, all points will be plotted.
        endpoints: bool
          If True, plot only the start and end points of the trajectory.
        sample: int, optional
          If specified, randomly sample this many particles to plot.
        earth: bool
          If True, draw a circle representing the Earth at the origin.
        grid: bool
          If True, add a grid to the plot.
        title: str
          Title of the plot.

        Returns
        -------
        The axis with the plotted trajectory.

        Examples
        --------
        >>> hist = disco.ParticleHistory.load("particle_history.h5")
        >>> hist.plot_xy()
        >>> plt.savefig('myplot.png')
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

        Parameters
        ----------
        ax: matplotlib axes
          Matplotlib axis to plot on. If None, a new figure and axis will be created.
        inds: int or list of ints
          Indices of the points to plot. If None, all points will be plotted.
        endpoints: bool
          If True, plot only the start and end points of the trajectory.
        sample: int, optional
          If specified, randomly sample this many particles to plot.
        earth: bool
          If True, draw a circle representing the Earth at the origin.
        grid: bool
          If True, add a grid to the plot.
        title: str
          Title of the plot.

        Returns
        -------
        The axis with the plotted trajectory.

        Examples
        --------
        >>> hist = disco.ParticleHistory.load("particle_history.h5")
        >>> hist.plot_xz()
        >>> plt.savefig('myplot.png')
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

        Parameters
        ----------
        ax: matplotlib axes
          Matplotlib axis to plot on. If None, a new figure and axis will be created.
        inds: int or list of ints
          Indices of the points to plot. If None, all points will be plotted.
        endpoints: bool
          If True, plot only the start and end points of the trajectory.
        sample: int, optional
          If specified, randomly sample this many particles to plot.
        earth: bool
          If True, draw a circle representing the Earth at the origin.
        grid: bool
          If True, add a grid to the plot.
        title: str
          Title of the plot.

        Returns
        -------
        The axis with the plotted trajectory.

        Examples
        --------
        >>> hist = disco.ParticleHistory.load("particle_history.h5")
        >>> hist.plot_yz()
        >>> plt.savefig('myplot.png')
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


class ParticleHistoryBuffer:
    """Buffer for storing history of particle trajectories.

    This class is used to accumulate history of particle trajectories
    during the tracing process, and related variables tracked.
    """

    def __init__(self):
        """Initialize a `HistoryBuffer` instance."""
        self.t = []
        self.y = []
        self.ppar = []
        self.B = []
        self.W = []
        self.h = []
        self.stopped = []
        self.extra_fields = []

    def append(self, t, y, B, h, stopped, extra_fields, total_reorder=None):
        """Append a new history entry to the buffer."""
        if total_reorder is None:
            total_reorder_rev = np.arange(len(t), dtype=int)
        else:
            total_reorder_rev = np.argsort(total_reorder)

        _, W = _calc_gamma_W(B, y)

        self.t.append(t[total_reorder_rev].get())
        self.y.append(y[total_reorder_rev].get())
        self.B.append(B[total_reorder_rev].get())
        self.W.append(W[total_reorder_rev].get())
        self.h.append(h[total_reorder_rev].get())
        self.stopped.append(stopped[total_reorder_rev].get())

        if extra_fields.size > 0:
            self.extra_fields.append(extra_fields[total_reorder_rev].get())

    def to_particle_history(self, particle_state, field_model):
        """Convert the accumulated history to a `ParticleHistory` instance.

        Parameters
        ----------
        particle_state: `ParticleState`
            The initial conditions of the particles, used to set mass and charge.

        Returns
        -------
        ParticleHistory
            A `ParticleHistory` instance containing the accumulated history, which
            can be used to save or plot results.
        """
        hist_t = undim_time(np.array(self.t))
        hist_B = undim_magnetic_field(np.array(self.B), particle_state.mass, particle_state.charge)
        hist_W = undim_energy(np.array(self.W), particle_state.mass)
        hist_h = undim_time(np.array(self.h))
        hist_stopped = np.array(self.stopped)
        hist_raw_extra_fields = np.array(self.extra_fields)
        hist_y = np.array(self.y)
        hist_pos_x = undim_space(hist_y[:, :, 0])
        hist_pos_y = undim_space(hist_y[:, :, 1])
        hist_pos_z = undim_space(hist_y[:, :, 2])
        hist_ppar = undim_momentum(hist_y[:, :, 3], particle_state.mass)
        hist_M = undim_magnetic_moment(hist_y[:, :, 4], particle_state.charge)

        if len(field_model.extra_fields) > 0:
            hist_extra_fields = {}
            key_names = list(field_model.extra_fields.keys())

            for i, key in enumerate(key_names):
                hist_extra_fields[key] = hist_raw_extra_fields[:, :, i]
        else:
            hist_extra_fields = None

        return ParticleHistory(
            t=hist_t,
            x=hist_pos_x,
            y=hist_pos_y,
            z=hist_pos_z,
            ppar=hist_ppar,
            M=hist_M,
            B=hist_B,
            W=hist_W,
            h=hist_h,
            stopped=hist_stopped,
            mass=particle_state.mass,
            charge=particle_state.charge,
            extra_fields=hist_extra_fields,
        )


def _calc_gamma_W(B, y):
    """Calculate  gamma (relativistic factor) and W (relativistic energy) for
    saving in history.

    Parameters
    ----------
    B : cupy array
       Magnetic Field Strength, dimensionalized
    y : cupy array
       State vector, dimensionalied

    Returns
    -------
    gamma: cupy array
       Relativstic factor, dimensionalized
    W : cupy array
       Relativistic Energy, dimensionalized
    """
    gamma = cp.sqrt(1 + 2 * B * y[:, 4] + y[:, 3] ** 2)
    W = gamma - 1
    return gamma, W
