"""Classes to load field models from simulation output and files on disk."""
from datetime import datetime
import glob
import os

from astropy import constants, units
import h5py
import numpy as np
from spacepy import pycdf

from disco import Axes
from disco.constants import DEFAULT_B0
from disco._regrid import regrid_pointcloud
from disco._field_model import FieldModel
from disco._field_model_loader import FieldModelLoader, LazyFieldModelLoader, StaticFieldModelLoader


class InvalidReaderShapeError(Exception):
    """Raised when an array in a file does not have the expected shape
    of number of dimensions.
    """


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
        B0=DEFAULT_B0,
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
        B0: quantity, units of magnetic field strength
            Internal model to use when computing the electric field from -uxB

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
        self.cdf_files = glob.glob(glob_pattern)
        self.cdf_files.sort()

        # Check all files are CDF files
        for cdf_file in self.cdf_files:
            if not cdf_file.lower().endswith(".cdf"):
                raise ValueError(f"Passed glob pattern that includes non-cdf file {repr(cdf_file)}")

        if not len(self.cdf_files) > 0:
            raise FileNotFoundError(
                "Glob pattern did not match any files."
            )

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

    def __getitem__(self, index):
        """Return field model for current index of the simulation dataset.

        Parameters
        ----------
        index: timestamp index, >= 0, and less then len(self)

        Returns
        -------
        field_model: instance of FieldModel with one timestep
        """
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

        #cdf.close()

        # Load Electric field as pointcloud
        Ex, Ey, Ez = -np.cross([ux.value, uy.value, uz.value], [Bx.value, By.value, Bz.value], axis=0)

        E_units = Bx.unit * ux.unit
        Ex *= E_units
        Ey *= E_units
        Ez *= E_units

        better_units = units.nV / units.m
        Ex = Ex.to(better_units)
        Ey = Ey.to(better_units)
        Ez = Ez.to(better_units)

        # Perform regridding
        field_model = regrid_pointcloud(
            x.value,
            y.value,
            z.value,
            self.time_axis[index].value,
            Bx_external,
            By_external,
            Bz_external,
            Ex.value,
            Ey.value,
            Ez.value,
            B0=self.B0,
        )

        return field_model


class GenericHdf5FieldModel(FieldModel):
    """Create a field model from values loaded out of an HDF5 file.

    Requires HDF5 variables:
    * `Bx`: shape (nx, ny, nz, nt), units nT
    * `By`: shape (nx, ny, nz, nt), units nT
    * `Bz`: shape (nx, ny, nz, nt), units nT
    * `Ex`: shape (nx, ny, nz, nt), units mV/m
    * `Ey`: shape (nx, ny, nz, nt), units mV/m
    * `Ez`: shape (nx, ny, nz, nt), units mV/m
    * `xaxis`: shape (nx,), units Re
    * `yaxis`: shape (ny,), units Re
    * `zaxis`: shape (nz,), units Re
    * `taxis`: shape (nt,), units seconds
    * `r_inner`: scalar, units Re

    Notes
    * B values must have the dipole subtracted (also known as
      being the "external" model)
    * Any NaN's encoutered will halt integration immediately,
      so they can be used to place irregular outer boundaries.
    """

    def __init__(self, hdf5_path, B0=DEFAULT_B0):
        """Load a `FieldModel` from a HDF5 file.

        Parameters
        ----------
        hdf_path: str
           Path to HDF5 File
        B0: Quantity, units of magnetic field strength
           Dipole intensity
        """
        # Read data from HDF5 file
        hdf_file = h5py.File(hdf5_path, "r")

        Bx = hdf_file["Bx"][:] * units.nT
        By = hdf_file["By"][:] * units.nT
        Bz = hdf_file["Bz"][:] * units.nT
        Ex = hdf_file["Ex"][:] * units.mV / units.m
        Ey = hdf_file["Ey"][:] * units.mV / units.m
        Ez = hdf_file["Ez"][:] * units.mV / units.m
        xaxis = hdf_file["xaxis"][:] * units.R_earth
        yaxis = hdf_file["yaxis"][:] * units.R_earth
        zaxis = hdf_file["zaxis"][:] * units.R_earth
        taxis = hdf_file["taxis"][:] * units.s
        r_inner = hdf_file["r_inner"][()] * units.R_earth

        hdf_file.close()

        # Check shapes
        self._check_shapes(Bx, By, Bz, Ex, Ey, Ez, xaxis, yaxis, zaxis, taxis)

        # Create axes instance
        axes = Axes(xaxis, yaxis, zaxis, taxis, r_inner)

        # Use parent constructor
        super().__init__(Bx, By, Bz, Ex, Ey, Ez, axes, B0=B0)

    def _check_shapes(self, Bx, By, Bz, Ex, Ey, Ez, xaxis, yaxis, zaxis, taxis):
        """Check that the shapes of the arrays match expected dimensions.

        Raises
        ------
        InvalidReaderShapeError
            If any array does not have the expected shap or number of dimensions.
        """
        # The the ndims of the 1D axis arrays
        expected_ndims = 1
        arrays = [xaxis, yaxis, zaxis, taxis]
        array_names = ["xaxis", "yaxis", "zaxis", "taxis"]

        for arr, name in zip(arrays, array_names):
            if len(arr.shape) != expected_ndims:
                raise InvalidReaderShapeError(
                    f"{name} has shape {arr.shape}, expected {expected_ndims} dimsensions"
                )

        # Check the shape of the 4D field arrays
        expected_shape = (xaxis.shape[0], yaxis.shape[0], zaxis.shape[0], taxis.shape[0])
        arrays = [Bx, By, Bz, Ex, Ey, Ez]
        array_names = ["Bx", "By", "Bz", "Ex", "Ey", "Ez"]

        for arr, name in zip(arrays, array_names):
            if arr.shape != expected_shape:
                raise InvalidReaderShapeError(
                    f"{name} has shape {arr.shape}, expected {expected_shape}" " (nx, ny, nz, nt)"
                )
