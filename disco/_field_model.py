"""Module for just the FieldModel and DimensionalizedFieldModel classes"""
from dataclasses import dataclass
import math
from typing import Dict

import cupy as cp
import h5py
import numpy as np

from disco import constants
from disco._axes import Axes
from disco._dimensionalization import dim_magnetic_field, dim_electric_field
from disco._kernels import multi_interp_kernel
from disco.constants import BLOCK_SIZE, DEFAULT_B0

from astropy import units, constants


class FieldModel:
    """Magnetic and electric field models used to propagate particles."""

    def __init__(self, Bx, By, Bz, Ex, Ey, Ez, axes, B0=DEFAULT_B0, extra_fields=None):
        """Get an instance that is dimensionalized and stored on the GPU.

        Input argument should have astropy units attached.

        Parameters
        ----------
        Bx: array of shape (nx, ny, nz, nt), with units
          External Magnetic Field X component, SM Coordinates
        By: array of shape (nx, ny, nz, nt), with units
          External Magnetic Field Y component, SM Coordinates
        Bz: array of shape (nx, ny, nz, nt), with units
          External Magnetic Field Z component, SM Coordinates
        Ex: array of shape (nx, ny, nz, nt), with units
          Electric Field X component, SM Coordinates
        Ey: array of shape (nx, ny, nz, nt), with units
          Electric Field Y component, SM Coordinates
        Ez: array of shape (nx, ny, nz, nt), with units
          Electric Field Z component, SM Coordinates
        axes: Axes
          Rectilinear grid information and inner boundary
        B0: scalar with units, optional
          Internal Dipole strength to add to external field interpolating.
        extra_fields: dict, optional
          Dictionary of additional fields to include in the field model, such as
          plasma parameters rho, p, T, etc. The keys should be strings representing
          the field names, and the values should be arrays with shape
          (nx, ny, nz, nt) with no units.
        """
        # Check that units are valid to catch errors early
        Bx.to(units.nT)
        By.to(units.nT)
        Bz.to(units.nT)
        # Check that units are valid to catch errors early
        Ex.to(units.mV / units.m)
        Ey.to(units.mV / units.m)
        Ez.to(units.mV / units.m)

        self.Bx = Bx
        self.By = By
        self.Bz = Bz
        self.Ex = Ex
        self.Ey = Ey
        self.Ez = Ez
        self.B0 = B0
        self.extra_fields = extra_fields if extra_fields is not None else {}
        self.axes = axes
        self.dimensionalized = False

    def duplicate_in_time(self, time_axis=[-1, 1] * units.year):
        """Duplicate the `FieldModel` in time, to support tracing in a single time step.

        Parameters
        ----------
        time_axis: array with units of time
          Time axis values to use for duplication. The default is made
          sufficiently large that it is unrealistic for a particle to
          ever hit the time limit bounds.

        Returns
        -------
        instance of FieldModel with duplicated time axis
        """

        nx, ny, nz, nt = self.Bx.shape

        if nt != 1:
            raise ValueError("Field model must have exactly one time step to duplicate in time.")

        # Duplicate main fields in time
        new_shape = (nx, ny, nz, 2)
        Bx = np.zeros(new_shape) + self.Bx
        By = np.zeros(new_shape) + self.By
        Bz = np.zeros(new_shape) + self.Bz
        Ex = np.zeros(new_shape) + self.Ex
        Ey = np.zeros(new_shape) + self.Ey
        Ez = np.zeros(new_shape) + self.Ez

        # Duplicate the extra fields in time
        new_extra_fields = {}

        for key, value in self.extra_fields.items():
            if value.shape != (nx, ny, nz, nt):
                raise ValueError(
                    f"Extra field '{key}' must have shape {(nx, ny, nz, nt)} to duplicate in time."
                )
            new_value = np.zeros(new_shape) + value
            new_extra_fields[key] = new_value

        # Create new axes with duplicated time axis
        axes = Axes(
            x=self.axes.x,
            y=self.axes.y,
            z=self.axes.z,
            t=time_axis,
            r_inner=self.axes.r_inner,
        )
        return FieldModel(Bx, By, Bz, Ex, Ey, Ez, axes, self.B0, extra_fields=new_extra_fields)

    def dimensionalize(self, mass, charge):
        """Convert to a `DimensionalizedFieldModel` instance.

        Parameters
        ----------
        mass : scalar with untis
          Scalar mass, used for dimensionalization
        charge : scalar with untis
           Scalar charge, used for dimensionalization

        Return
        ------
        instance of `DimensionalizedFieldModel`
        """
        return DimensionalizedFieldModel(self, mass, charge)

    def save(self, hdf_path):
        """Save the `FieldModel` to an HDF5 file. Can be loaded with the `FieldModel.load()` method.

        Parameters
        ----------
        hdf5_path: str
            Path to the HDF5 file to save to.
        """
        B_units = units.nT
        E_units = units.mV / units.m
        space_units = constants.R_earth
        time_units = units.s

        with h5py.File(hdf_path, "w") as hdf_file:
            hdf_file.create_dataset("Bx", data=self.Bx.to(B_units).value)
            hdf_file.create_dataset("By", data=self.By.to(B_units).value)
            hdf_file.create_dataset("Bz", data=self.Bz.to(B_units).value)
            hdf_file.create_dataset("Ex", data=self.Ex.to(E_units).value)
            hdf_file.create_dataset("Ey", data=self.Ey.to(E_units).value)
            hdf_file.create_dataset("Ez", data=self.Ez.to(E_units).value)
            hdf_file.create_dataset("xaxis", data=self.axes.x.to(space_units).value)
            hdf_file.create_dataset("yaxis", data=self.axes.y.to(space_units).value)
            hdf_file.create_dataset("zaxis", data=self.axes.z.to(space_units).value)
            hdf_file.create_dataset("taxis", data=self.axes.t.to(time_units).value)

            # Save r_inner and B0 as a scalar
            hdf_file.create_dataset("r_inner", data=self.axes.r_inner.to(space_units).value)

            # Save extra fields
            if self.extra_fields:
                extra_fields_group = hdf_file.create_group("extra_fields")

                for key, value in self.extra_fields.items():
                    extra_fields_group.create_dataset(key, data=value)

    @classmethod
    def load(cls, hdf_path, B0=DEFAULT_B0):
        """Load a FieldModel from an HDF5 file.

        Parameters
        ----------
        hdf5_path: str
            Path to the HDF5 file to load from.
        B0: scalar with units, optional
          Internal Dipole strength to add to external field interpolating.

        Returns
        -------
        `FieldModel`
            An instance of `FieldModel` loaded from the HDF5 file.

        Raises
        ------
        `ValueError`
            If the shapes of the arrays in the HDF5 file do not match expected dimensions.

        Notes
        -----
        Required HDF5 variables:

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

        Additional comments:

        * B values must have the dipole subtracted (also known as
          being the "external" model)

        * Any NaN's encoutered will halt integration immediately,
          so they can be used to place irregular outer boundaries.
        """
        # Read data from HDF5 file
        hdf_file = h5py.File(hdf_path, "r")

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

        extra_fields = {}

        if "extra_fields" in hdf_file.keys():
            extra_fields_group = hdf_file["extra_fields"]

            for key, value in extra_fields_group.items():
                extra_fields[key] = value[:]  # No units, just raw data

        hdf_file.close()

        # Check shapes
        cls._check_shapes(Bx, By, Bz, Ex, Ey, Ez, xaxis, yaxis, zaxis, taxis)

        # Create axes instance
        axes = Axes(xaxis, yaxis, zaxis, taxis, r_inner)

        # Use class constructor
        return cls(Bx, By, Bz, Ex, Ey, Ez, axes, B0=B0, extra_fields=extra_fields)

    @classmethod
    def _check_shapes(cls, Bx, By, Bz, Ex, Ey, Ez, xaxis, yaxis, zaxis, taxis):
        """Check that the shapes of the arrays match expected dimensions.

        Raises
        ------
        ValueError
            If any array does not have the expected shap or number of dimensions.
        """
        # The the ndims of the 1D axis arrays
        expected_ndims = 1
        arrays = [xaxis, yaxis, zaxis, taxis]
        array_names = ["xaxis", "yaxis", "zaxis", "taxis"]

        for arr, name in zip(arrays, array_names):
            if len(arr.shape) != expected_ndims:
                raise ValueError(
                    f"{name} has shape {arr.shape}, expected {expected_ndims} dimsensions"
                )

        # Check the shape of the 4D field arrays
        expected_shape = (xaxis.shape[0], yaxis.shape[0], zaxis.shape[0], taxis.shape[0])
        arrays = [Bx, By, Bz, Ex, Ey, Ez]
        array_names = ["Bx", "By", "Bz", "Ex", "Ey", "Ez"]

        for arr, name in zip(arrays, array_names):
            if arr.shape != expected_shape:
                raise ValueError(
                    f"{name} has shape {arr.shape}, expected {expected_shape}" " (nx, ny, nz, nt)"
                )


class DimensionalizedFieldModel:
    """Dimensionalized magnetic and electric field models
    used to propagate particles
    ."""

    def __init__(self, field_model, mass, charge):
        """Get an instance that is dimensionalized and stored on the GPU.

        mass is not part of field model, but is used to redimensionalize.
        Input argument should have astropy units attached.

        Warning
        --------
        This class is not threadsafe.

        Parameters
        ----------
        field_model: instance of FieldModel
        mass: mass of particle, with units
        charge: charge of particle, with units
        B0: dipole strength
        """
        self.negative_charge = charge.value < 0
        self.Bx = cp.array(dim_magnetic_field(field_model.Bx, mass, charge))
        self.By = cp.array(dim_magnetic_field(field_model.By, mass, charge))
        self.Bz = cp.array(dim_magnetic_field(field_model.Bz, mass, charge))
        self.B0 = cp.array(dim_magnetic_field(field_model.B0, mass, charge))
        self.Ex = cp.array(dim_electric_field(field_model.Ex, mass, charge))
        self.Ey = cp.array(dim_electric_field(field_model.Ey, mass, charge))
        self.Ez = cp.array(dim_electric_field(field_model.Ez, mass, charge))
        self.extra_fields = {}

        for key, value in field_model.extra_fields.items():
            self.extra_fields[key] = cp.array(value)

        self.axes = field_model.axes.dimensionalize()

        self._memory_initialized = False
        self._memory_arr_size = None

    def _memory_initialize(self, arr_size):
        """
        Initializes variables reused in each multi_interp() call. They
        are defined once as an optimizatin. See warnings in
        the class header and multi_interp()
        """
        # Only needs to be done once unless the arr_size changes
        if self._memory_initialized and self._memory_arr_size == arr_size:
            return

        # Initialize variables
        nx = self.axes.x.size
        ny = self.axes.y.size
        nz = self.axes.z.size
        nt = self.axes.t.size
        nttl = nx * ny * nz * nt

        self._B0_arr = cp.zeros(arr_size) + self.B0
        self._r_inner = cp.zeros(arr_size) + self.axes.r_inner
        self._Bxvec = self.Bx.reshape(nttl, order="F")
        self._Byvec = self.By.reshape(nttl, order="F")
        self._Bzvec = self.Bz.reshape(nttl, order="F")
        self._Exvec = self.Ex.reshape(nttl, order="F")
        self._Eyvec = self.Ey.reshape(nttl, order="F")
        self._Ezvec = self.Ez.reshape(nttl, order="F")
        self._extra_fields_vec = cp.zeros((len(self.extra_fields), arr_size))

        for i, value in enumerate(self.extra_fields.values()):
            self._extra_fields_vec[i, :] = value.reshape(nttl, order="F")

        self._Bx_out = cp.zeros(arr_size)
        self._By_out = cp.zeros(arr_size)
        self._Bz_out = cp.zeros(arr_size)
        self._Ex_out = cp.zeros(arr_size)
        self._Ey_out = cp.zeros(arr_size)
        self._Ez_out = cp.zeros(arr_size)
        self._dBxdx_out = cp.zeros(arr_size)
        self._dBxdy_out = cp.zeros(arr_size)
        self._dBxdz_out = cp.zeros(arr_size)
        self._dBydx_out = cp.zeros(arr_size)
        self._dBydy_out = cp.zeros(arr_size)
        self._dBydz_out = cp.zeros(arr_size)
        self._dBzdx_out = cp.zeros(arr_size)
        self._dBzdy_out = cp.zeros(arr_size)
        self._dBzdz_out = cp.zeros(arr_size)
        self._B_out = cp.zeros(arr_size)
        self._dBdx_out = cp.zeros(arr_size)
        self._dBdy_out = cp.zeros(arr_size)
        self._dBdz_out = cp.zeros(arr_size)
        self._extra_fields_out = cp.zeros((len(self.extra_fields), arr_size))

        # Save state variables
        self._memory_initialized = True
        self._memory_arr_size = arr_size

    def multi_interp(self, t, y, paused, stopped_cutoff):
        """Interpolate field values at given positions.

        Warning
        -------
        The CuPy arrays in the output arrays are atttached
        to this instance. Subsequent calls will overwrite
        them. If you want to avoid this, call .copy() on
        each array.

        Paramaters
        ----------
        t: cupy array
          Vector of dimensionalized particle times
        y: cupy array
           Vector of shape (npart, nstate) of particle states
        paused: cupy array
            Boolean array indicating whether interpolation was not done
            because the required time slice of field data is not loaded.
        stopped_cutoff: int
           Cutoff index for partiles that no longer require processing

        Returns
        -------
        Instance of _MultiInterpResult

        Raises
        ------
        RuntimeError: Interpolation requires at minimum two timesteps
        """
        # Make sure conditoins of quadlinear interpolation are met
        if self.axes.t.size < 2:
            raise RuntimeError("Field model interpolation requires at minimum two timesteps")

        # Use Axes object to get neighbors of cell
        neighbors = self.axes.get_neighbors(t, y)

        # Setup variables to send to GPU kernel
        arr_size = y.shape[0]

        self._memory_initialize(arr_size)

        nx = self.axes.x.size
        ny = self.axes.y.size
        nz = self.axes.z.size
        nt = self.axes.t.size
        nxy = nx * ny
        nxyz = nxy * nz
        nttl = nxyz * nt

        x_axis = self.axes.x
        y_axis = self.axes.y
        z_axis = self.axes.z
        t_axis = self.axes.t

        ix, iy, iz, it = (
            neighbors.field_i,
            neighbors.field_j,
            neighbors.field_k,
            neighbors.field_l,
        )

        # Call GPU Kernel
        grid_size = int(math.ceil(stopped_cutoff / BLOCK_SIZE))

        multi_interp_kernel[grid_size, BLOCK_SIZE](
            nx,
            ny,
            nz,
            nt,
            nxy,
            nxyz,
            nttl,
            ix,
            iy,
            iz,
            it,
            t,
            y[:stopped_cutoff],
            paused,
            self._B0_arr,
            self._r_inner,
            t_axis,
            x_axis,
            y_axis,
            z_axis,
            self._Bxvec,
            self._Byvec,
            self._Bzvec,
            self._Exvec,
            self._Eyvec,
            self._Ezvec,
            self._extra_fields_vec,
            self._Bx_out,
            self._By_out,
            self._Bz_out,
            self._Ex_out,
            self._Ey_out,
            self._Ez_out,
            self._dBxdx_out,
            self._dBxdy_out,
            self._dBxdz_out,
            self._dBydx_out,
            self._dBydy_out,
            self._dBydz_out,
            self._dBzdx_out,
            self._dBzdy_out,
            self._dBzdz_out,
            self._B_out,
            self._dBdx_out,
            self._dBdy_out,
            self._dBdz_out,
            self._extra_fields_out,
        )

        # need to account for dimensionalization of magnitude
        if self.negative_charge:
            self._B_out *= -1
            self._dBdx_out *= -1
            self._dBdy_out *= -1
            self._dBdz_out *= -1

        extra_fields_out_dict = {}

        for i, (key, value) in enumerate(self.extra_fields.items()):
            # Add to the output dictionary
            extra_fields_out_dict[key] = self._extra_fields_out[i, :]

        # Return values as tuple
        return _MultiInterpResult(
            Bx=self._Bx_out,
            By=self._By_out,
            Bz=self._Bz_out,
            Ex=self._Ex_out,
            Ey=self._Ey_out,
            Ez=self._Ez_out,
            dBxdx=self._dBxdx_out,
            dBxdy=self._dBxdy_out,
            dBxdz=self._dBxdz_out,
            dBydx=self._dBydx_out,
            dBydy=self._dBydy_out,
            dBydz=self._dBydz_out,
            dBzdx=self._dBzdx_out,
            dBzdy=self._dBzdy_out,
            dBzdz=self._dBzdz_out,
            B=self._B_out,
            dBdx=self._dBdx_out,
            dBdy=self._dBdy_out,
            dBdz=self._dBdz_out,
            extra_fields=self._extra_fields_out,
        )


@dataclass
class _MultiInterpResult:
    """Data container for return value of
    _DimensionalizedFieldModel.multi_interp()
    """

    # Field values and partial derivatives at the requested points.
    Bx: cp.array
    By: cp.array
    Bz: cp.array
    Ex: cp.array
    Ey: cp.array
    Ez: cp.array
    dBxdx: cp.array
    dBxdy: cp.array
    dBxdz: cp.array
    dBydx: cp.array
    dBydy: cp.array
    dBydz: cp.array
    dBzdx: cp.array
    dBzdy: cp.array
    dBzdz: cp.array
    B: cp.array
    dBdx: cp.array
    dBdy: cp.array
    dBdz: cp.array
    extra_fields: Dict[str, cp.array]
