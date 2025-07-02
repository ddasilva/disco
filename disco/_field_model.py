"""Module for just the FieldModel and DimensionalizedFieldModel classes"""
from dataclasses import dataclass
import math

import cupy as cp

from disco._dimensionalization import dim_magnetic_field, dim_electric_field
from disco._kernels import multi_interp_kernel
from disco.constants import BLOCK_SIZE, DEFAULT_B0

from astropy import units


class FieldModel:
    """Magnetic and electric field models used to propagate particles."""

    def __init__(self, Bx, By, Bz, Ex, Ey, Ez, axes, B0=DEFAULT_B0):
        """Get an instance that is dimensionalized and stored on the GPU.

        mass is not part of field model, but is used to redimensionalize.
        Input argument should have astropy units attached.

        Parameters
        ----------
        Bx: array of shape (nx, ny, nz, nt), with units
          External Magnetic Field X component
        By: array of shape (nx, ny, nz, nt), with units
          External Magnetic Field Y component
        Bz: array of shape (nx, ny, nz, nt), with units
          External Magnetic Field Z component
        Ex: array of shape (nx, ny, nz, nt), with units
          Electric Field X component
        Ey: array of shape (nx, ny, nz, nt), with units
          Electric Field Y component
        Ez: array of shape (nx, ny, nz, nt), with units
          Electric Field Z component
        axes: Axes
          Grid information
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
        self.Bz = Bx
        self.Ex = Ex
        self.Ey = Ey
        self.Ez = Ez
        self.B0 = B0
        self.axes = axes
        self.dimensionalized = False

    def dimensionalize(self, mass, charge):
        """Convert to a `DimensionalizedFieldModel` instance.

        Parameters
        ----------
        mass : Quantity
          Scalar mass, used for dimensionalization
        charge : Quantity
           Scalar charge, used for dimensionalization

        Return
        ------
        instance of `DimensionalizedFieldModel`
        """
        return DimensionalizedFieldModel(self, mass, charge)


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
           Vector of shape (npart, 5) of particle states
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
        )

        # need to account for dimensionalization of magnitude
        if self.negative_charge:
            self._B_out *= -1
            self._dBdx_out *= -1
            self._dBdy_out *= -1
            self._dBdz_out *= -1

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
