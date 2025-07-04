"""Classes to load field models from simulation output and files on disk."""
from datetime import datetime
import glob
import os
import time

from astropy import constants, units
import h5py
import numpy as np
from spacepy import pycdf

from disco import Axes
from disco.constants import DEFAULT_B0
from disco._regrid import regrid_pointcloud
from disco._field_model import FieldModel
from disco._field_model_loader import FieldModelLoader, LazyFieldModelLoader, StaticFieldModelLoader


class FieldModelDataset:
    """This is an abstract base class to provide lazy loading for simulation
    output.

    See also
    --------
    disco.LazyFieldModelLoader
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


class SwmfCdfFieldModelDataset(FieldModelDataset):
    """Subclass of FieldModelDataset for lazy loading of SWMF CDF Output"""

    def __init__(
        self,
        glob_pattern,
        timestamp_parser="3d__var_1_e%Y%m%d-%H%M%S",
        timestamp_trim=12,
        t0=None,
        cache_regrid=True,
        cache_regrid_dir="same_dir",
        B0=DEFAULT_B0,
        r_inner=2.5 * constants.R_earth,
        verbose=1,
        grid_downsample=2,
    ):
        """Create an instance of SwmfCdfFieldModelDataset

        Parameters
        ----------
        glob_pattern: str
            Pattern such as "/home/ubuntu/simulation_output/*.cdf"
        timestamp_parser: str
            datetime strptime pattern for parsing timestamps from filenames
        timestamp_trim: int
            trim this many characters from end of filenames before parsing timestamp
        t0: datetime, optional
            If provided, the timestamps will be relative to this time.
            If not provided, the first timestamp in the dataset will be used as t0.
        cache_regrid: bool
            If True, the regridded data will be cached on disk for faster access
            If False, the regridding will be done every time __getitem__ is called.
        cache_regrid_dir: str
            Directory to cache regridded data. If 'same_dir', it will use the same directory as
            the CDF files.
        B0: quantity, units of magnetic field strength
            Internal model to use when computing the electric field from -uxB
        verbose: int
            Verbosity level for output. Set to 0 for no output

        Raises
        ------
        ValueError
            If the glob pattern matches non-CDF files.
        FileNotFoundError
            If the glob pattern does not match at least 2 CDF files.

        Notes
        -----
        The glob pattern must match at least one CDF files, and all files must
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
            # Use the same directory as the CDF files for caching
            self.cache_regrid_dir = os.path.dirname(glob_pattern)
        else:
            self.cache_regrid_dir = cache_regrid_dir

        # Get all CDF files matching the glob pattern
        self.cdf_files = glob.glob(glob_pattern)
        self.cdf_files.sort()

        # Check all files are CDF files
        for cdf_file in self.cdf_files:
            if not cdf_file.lower().endswith(".cdf"):
                raise ValueError(f"Passed glob pattern that includes non-cdf file {repr(cdf_file)}")

        if not len(self.cdf_files) > 0:
            raise FileNotFoundError("Glob pattern did not match any files.")

        # Get timestamps as datetime from the file names
        self.timestamps = []

        for cdf_file in self.cdf_files:
            time_str = os.path.basename(cdf_file[:-timestamp_trim])
            timestamp = datetime.strptime(time_str, timestamp_parser)
            self.timestamps.append(timestamp)

        # Precompute time axis
        self.time_axis = []
        if t0 is None:
            t0 = self.timestamps[0]

        for timestamp in self.timestamps:
            time = (timestamp - t0).total_seconds()
            self.time_axis.append(time)

        self.time_axis = np.array(self.time_axis) * units.s

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
        return len(self.cdf_files)

    def get_cache_file(self, index):
        """Get the cache file path for the given index.

        Parameters
        ----------
        index: int
            Index of the CDF file to get the cache file for.

        Returns
        -------
        cache_file: str
            Path to the cache file.
        """
        return os.path.join(
            self.cache_regrid_dir,
            os.path.basename(self.cdf_files[index]).replace(".cdf", "_regrid.h5"),
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
        cdf = pycdf.CDF(self.cdf_files[index])

        # Load XYZ Positoins
        x = cdf["x"][:].squeeze() * constants.R_earth
        y = cdf["y"][:].squeeze() * constants.R_earth
        z = cdf["z"][:].squeeze() * constants.R_earth

        # Load Magnetic Field as pointcloud
        Bx_external = cdf["bx"][:].squeeze() * units.nT
        By_external = cdf["by"][:].squeeze() * units.nT
        Bz_external = cdf["bz"][:].squeeze() * units.nT

        r = np.sqrt(x**2 + y**2 + z**2).value
        Bx_dipole = 3 * x.value * z.value * self.B0 / r**5
        By_dipole = 3 * y.value * z.value * self.B0 / r**5
        Bz_dipole = (3 * z.value**2 - r**2) * self.B0 / r**5

        Bx = Bx_dipole + Bx_external
        By = By_dipole + By_external
        Bz = Bz_dipole + Bz_external

        # Load Flow Velocity as pointcloud
        ux = cdf["ux"][:].squeeze() * constants.R_earth / units.s
        uy = cdf["uy"][:].squeeze() * constants.R_earth / units.s
        uz = cdf["uz"][:].squeeze() * constants.R_earth / units.s

        cdf.close()

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
