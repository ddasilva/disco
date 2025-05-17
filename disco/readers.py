"""Classes to load FieldModel instances from various file formats.

Classes in this module are subclasses of `FieldModel`.
"""

from astropy import units
import h5py

from disco import Axes
from disco.constants import DEFAULT_B0
from disco._field_model import FieldModel


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

        # Create axes instance
        axes = Axes(xaxis, yaxis, zaxis, taxis, r_inner)

        # Use parent constructor
        super().__init__(Bx, By, Bz, Ex, Ey, Ez, axes, B0=B0)
