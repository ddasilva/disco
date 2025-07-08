"""Classes to load field models from simulation output and files on disk."""
from datetime import timedelta, datetime
import glob
import os
import time

from astropy import constants, units
import h5py
import numpy as np
from spacepy import pybats

from disco import Axes
from disco.constants import DEFAULT_B0
from disco._regrid import regrid_pointcloud
from disco._field_model import FieldModel
from disco._field_model_loader import FieldModelLoader, LazyFieldModelLoader, StaticFieldModelLoader


__all__ = [
    "FieldModelDataset",
    "SwmfOutFieldModelDataset",
]


class FieldModelDataset:
    """This is an abstract base class to provide lazy loading for simulation
    output.

    .. automethod:: __getitem__
    .. automethod:: __len__

    Notes
    -----
    See also: disco.LazyFieldModelLoader
    """

    def __init__(self):
        raise NotImplementedError()

    def get_time_axis(self):
        """Get time axis with length equal to len(self)

        Returns
        -------
        time_axis: array with units of time, and size equal to len(self)
        """
        raise NotImplementedError()

    def __len__(self):
        """Return length of the dataset in number of indices.

        Returns
        -------
        length: integer number of timesteps
        """
        raise NotImplementedError()

    def __getitem__(self, index):
        """Return field model for current index of the simulation dataset.

        Parameters
        ----------
        index: timestamp index, >= 0, and less then len(self)

        Returns
        -------
        field_model: instance of FieldModel with one timestep
        """
        raise NotImplementedError()


class SwmfOutFieldModelDataset(FieldModelDataset):
    """Subclass of FieldModelDataset for lazy reading of SWMF .out Output

    .. automethod:: __getitem__
    .. automethod:: __len__
    """

    def __init__(
        self,
        glob_pattern,
        t0=None,
        cache_regrid=True,
        cache_regrid_dir="same_dir",
        B0=DEFAULT_B0,
        r_inner=2.5 * constants.R_earth,
        verbose=1,
        grid_downsample=2,
    ):
        """Create an instance of SwmfOutFieldModelDataset

        Parameters
        ----------
        glob_pattern: str
            Pattern such as "/home/ubuntu/simulation_output/*.out"
        t0: datetime or scalar with time units
            Initial time for dataset. If filenames have a full date in them, pass a datetime
            If filenames have a relative time, pass a scalar with units of time.
            If None, the first timestamp in the globbed files will be used.
        cache_regrid: bool
            If True, the regridded data will be cached on disk for faster access
            If False, the regridding will be done every time __getitem__ is called.
        cache_regrid_dir: str
            Directory to cache regridded data. If 'same_dir', it will use the same directory as
            the CDF files.
        B0: quantity, units of magnetic field strength
            Internal model to use in returned `FieldModel` instances.
        verbose: int
            Verbosity level for output. Set to 0 for no output

        Raises
        ------
        ValueError
            If the glob pattern matches non .out files.
        FileNotFoundError
            If the glob pattern does not match any files.

        Notes
        -----
        The glob pattern must match at least one .out files, and all files must
        have the same timestamp format.

        The timestamps are parsed from the filenames, so they must be in a format
        that can be parsed by `datetime.strptime` with the provided `timestamp_parser`.
        """
        self.B0 = B0
        self.r_inner = r_inner
        self.grid_downsample = grid_downsample
        self.verbose = verbose
        self.cache_regrid = cache_regrid

        if cache_regrid_dir == "same_dir":
            # Use the same directory as the .out files for caching
            self.cache_regrid_dir = os.path.dirname(glob_pattern)
        else:
            self.cache_regrid_dir = cache_regrid_dir

        # Get all .out files matching the glob pattern
        self.out_files = glob.glob(glob_pattern)
        self.out_files.sort()

        # Check all files are .out files
        for out_file in self.out_files:
            if not out_file.lower().endswith(".out"):
                raise ValueError(f"Passed glob pattern that includes non-out file {repr(out_file)}")

        if not len(self.out_files) > 0:
            raise FileNotFoundError("Glob pattern did not match any files.")

        # Get timestamps as datetime from the file names
        if "_e" in os.path.basename(self.out_files[0]):
            self.time_axis = self._get_time_axis_absolute_times(t0)
        elif "_t" in os.path.basename(self.out_files[0]):
            self.time_axis = self._get_time_axis_relative_times(t0)
        else:
            raise ValueError(
                "Unable to determine timestamp format from file names. "
                "File names must contain '_e' for absolute times or '_t' for relative times."
            )

    def _get_time_axis_absolute_times(self, t0):
        """Parse absolute timestamps from file names and return time axis.

        Parameters
        ----------
        t0: datetime or None
            Initial time for the dataset. If None, the first timestamp in the files will be used.
            If a datetime object, it should be naive (no timezone info).

        Returns
        -------
        time_axis: array with units of time, and size equal to len(self)
        """
        timestamps = []

        for out_file in self.out_files:
            timestamp = None

            for tok in os.path.basename(out_file).split("_"):
                if tok.startswith("e") and len(tok) > 1:
                    # Extract the timestamp part
                    time_str = tok[1:]
                    timestamp = datetime.strptime(time_str[:-8], "%Y%m%d-%H%M%S")
                    break

            if timestamp is None:
                raise ValueError(f"Unable to parse timestamp from file name {out_file}")

            timestamps.append(timestamp)

        # Precompute time axis
        time_axis = []

        if t0 is None:
            t0 = timestamps[0]
        elif isinstance(t0, datetime):
            t0 = t0.replace(tzinfo=None)  # Ensure t0 is naive
        else:
            raise ValueError("t0 must be a datetime object or None.")

        for timestamp in timestamps:
            time = (timestamp - t0).total_seconds()
            time_axis.append(time)

        time_axis = np.array(time_axis) * units.s

        return time_axis

    def _get_time_axis_relative_times(self, t0):
        """Parse relative times from file timestamps and return time axis

        Parameters
        ----------
        t0: scalar with units of time or None
            Initial time for the dataset. If None, the first timestamp in the files will be used.
            If a scalar, it should have units of time (e.g., seconds).

        Returns
        -------
        time_axis: array with units of time, and size equal to len(self)
        """
        timestamps = []

        for out_file in self.out_files:
            timestamp = None

            for tok in os.path.basename(out_file).split("_"):
                if tok.startswith("t") and len(tok) > 1:
                    # Extract the timestamp part
                    time_str = tok[1:]
                    timestamp = timedelta(
                        days=int(time_str[:2]),
                        hours=int(time_str[2:4]),
                        minutes=int(time_str[4:6]),
                        seconds=int(time_str[6:8]),
                    )
                    break

            if timestamp is None:
                raise ValueError(f"Unable to parse timestamp from file name {out_file}")

            timestamps.append(timestamp)

        # Precompute time axis
        time_axis = []

        for timestamp in timestamps:
            time = timestamp.total_seconds()
            time_axis.append(time)

        time_axis = np.array(time_axis) * units.s

        if t0 is None:
            time_axis -= time_axis[0]
        elif isinstance(t0, units.Quantity):
            time_axis -= t0
        else:
            raise ValueError("t0 must be a scalar with units of time or None.")

        return time_axis

    def get_time_axis(self):
        """Get time axis with length equal to len(self)

        Returns
        -------
        time_axis: array with units of time, and size equal to len(self)
        """
        return self.time_axis

    def __len__(self):
        """Return length of the dataset in number of indices.

        Returns
        -------
        length: integer number of timesteps
        """
        return len(self.out_files)

    def get_cache_file(self, index):
        """Get the cache file path for the given index.

        Parameters
        ----------
        index: int
            Index of the file to get the cache file for.

        Returns
        -------
        cache_file: str
            Path to the cache file.
        """
        return os.path.join(
            self.cache_regrid_dir,
            os.path.basename(self.out_files[index]).replace(".out", "_disco.h5"),
        )

    def __getitem__(self, index):
        """Return field model for current index of the simulation dataset.

        Parameters
        ----------
        index: timestamp index, >= 0, and less then len(self)

        Returns
        -------
        field_model: instance of FieldModel with one timestep
        """
        # Use cached file is enabled and available
        cache_file = self.get_cache_file(index)

        if self.cache_regrid:
            if os.path.exists(cache_file):
                if self.verbose > 0:
                    print(f"Loading cached regridded data from {cache_file}")
                return FieldModel.load(cache_file)

        # Load CDF file
        out_file = pybats.IdlFile(self.out_files[index])

        # Load XYZ Positions
        x = out_file["x"][:].squeeze() * constants.R_earth
        y = out_file["y"][:].squeeze() * constants.R_earth
        z = out_file["z"][:].squeeze() * constants.R_earth

        # Load Magnetic Field as pointcloud
        Bx = out_file["bx"][:].squeeze() * units.nT
        By = out_file["by"][:].squeeze() * units.nT
        Bz = out_file["bz"][:].squeeze() * units.nT

        r = np.sqrt(x**2 + y**2 + z**2).value
        Bx_dipole = 3 * x.value * z.value * self.B0 / r**5
        By_dipole = 3 * y.value * z.value * self.B0 / r**5
        Bz_dipole = (3 * z.value**2 - r**2) * self.B0 / r**5

        Bx_external = Bx - Bx_dipole
        By_external = By - By_dipole
        Bz_external = Bz - Bz_dipole

        # Load Flow Velocity as pointcloud
        ux = out_file["ux"][:].squeeze() * units.km / units.s
        uy = out_file["uy"][:].squeeze() * units.km / units.s
        uz = out_file["uz"][:].squeeze() * units.km / units.s

        # Load Electric field as pointcloud
        Ex, Ey, Ez = -np.cross(
            [ux.value, uy.value, uz.value], [Bx.value, By.value, Bz.value], axis=0
        )

        E_units = Bx.unit * ux.unit
        Ex *= E_units
        Ey *= E_units
        Ez *= E_units

        better_E_units = units.mV / units.m

        # Perform regridding for B, using all points
        point_cloud_fields = {
            "Bx_external": Bx_external.to(units.nT).value,
            "By_external": By_external.to(units.nT).value,
            "Bz_external": Bz_external.to(units.nT).value,
            "Ex": Ex.to(better_E_units).value,
            "Ey": Ey.to(better_E_units).value,
            "Ez": Ez.to(better_E_units).value,
        }

        start_time = time.time()
        xaxis, yaxis, zaxis, regrid_fields = regrid_pointcloud(
            x.to(constants.R_earth).value,
            y.to(constants.R_earth).value,
            z.to(constants.R_earth).value,
            point_cloud_fields,
            grid_downsample=self.grid_downsample,
        )
        end_time = time.time()

        if self.verbose > 0:
            print(f"Regridding took {end_time - start_time:.2f} seconds")

        # Create field model and axes instances
        axes = Axes(
            xaxis * constants.R_earth,
            yaxis * constants.R_earth,
            zaxis * constants.R_earth,
            self.time_axis[[index]],
            r_inner=self.r_inner,
        )

        field_model = FieldModel(
            regrid_fields["Bx_external"] * units.nT,
            regrid_fields["By_external"] * units.nT,
            regrid_fields["Bz_external"] * units.nT,
            regrid_fields["Ex"] * better_E_units,
            regrid_fields["Ey"] * better_E_units,
            regrid_fields["Ez"] * better_E_units,
            axes,
            B0=self.B0,
        )

        # Save regridded data to cache if enabled
        if self.cache_regrid:
            field_model.save(cache_file)

        return field_model
