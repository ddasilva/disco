"""Module for just the FieldModel class, to prevent circular imports."""
from dataclasses import dataclass
import math

import cupy as cp

from disco._kernels import multi_interp_kernel
from disco.constants import BLOCK_SIZE, DEFAULT_B0

from astropy import units, constants


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
        Ex.to(units.mV/units.m)
        Ey.to(units.mV/units.m)
        Ez.to(units.mV/units.m)
        
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
    DEFAULT_RAW_B0 = 31e3 * units.nT

    def __init__(self, field_model, mass, charge):
        """Get an instance that is dimensionalized and stored on the GPU.

        mass is not part of field model, but is used to redimensionalize.
        Input argument should have astropy units attached.

        Parameters
        ----------
        field_model: instance of FieldModel
        mass: mass of particle, with units
        charge: charge of particle, with units
        B0: dipole strength
        """
        q = charge
        Re = constants.R_earth
        c = constants.c
        sf = q * Re / (mass * c**2)
        B_units = units.s / Re

        self.negative_charge = q.value < 0
        self.Bx = cp.array((sf * field_model.Bx).to(B_units).value)
        self.By = cp.array((sf * field_model.By).to(B_units).value)
        self.Bz = cp.array((sf * field_model.Bz).to(B_units).value)
        self.B0 = float((sf * field_model.B0).to(B_units).value)
        self.Ex = cp.array((sf * field_model.Ex).to(1).value)
        self.Ey = cp.array((sf * field_model.Ey).to(1).value)
        self.Ez = cp.array((sf * field_model.Ez).to(1).value)
        self.axes = field_model.axes
        
    def multi_interp(self, t, y, stopped_cutoff):
        """Interpolate field values at given positions.

        Paramaters
        ----------
        t: cupy array
          Vector of dimensionalized particle times
        y: cupy array
           Vector of shape (npart, 5) of particle states
        axes: Axes
           Set of axes to interpolate on
        stopped_cutoff: int
           Cutoff index for partiles that no longer require processing
        Returns
        -------
        Bx, By, Bz, Ex, Ey, Ez, dBxdx, dBxdy, dBxdz, dBydx, dBydy, dBydz,
        dBzdx, dBzdy, dBzdz, B, dBdx, dBdy, dBdz
        """
        # Use Axes object to get neighbors of cell
        neighbors = self.axes.get_neighbors(t, y[:, 0], y[:, 1], y[:, 2])

        # Setup variables to send to GPU kernel
        arr_size = y.shape[0]
        nx = self.axes.x.size
        ny = self.axes.y.size
        nz = self.axes.z.size
        nt = self.axes.t.size
        nxy = nx * ny
        nxyz = nxy * nz
        nttl = nxyz * nt

        B0 = cp.zeros(arr_size) + self.B0
        r_inner = cp.zeros(arr_size) + self.axes.r_inner

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

        Bxvec = self.Bx.reshape(nttl, order="F")
        Byvec = self.By.reshape(nttl, order="F")
        Bzvec = self.Bz.reshape(nttl, order="F")
        Exvec = self.Ex.reshape(nttl, order="F")
        Eyvec = self.Ey.reshape(nttl, order="F")
        Ezvec = self.Ez.reshape(nttl, order="F")

        Bx = cp.zeros(arr_size)
        By = cp.zeros(arr_size)
        Bz = cp.zeros(arr_size)
        Ex = cp.zeros(arr_size)
        Ey = cp.zeros(arr_size)
        Ez = cp.zeros(arr_size)
        dBxdx = cp.zeros(arr_size)
        dBxdy = cp.zeros(arr_size)
        dBxdz = cp.zeros(arr_size)
        dBydx = cp.zeros(arr_size)
        dBydy = cp.zeros(arr_size)
        dBydz = cp.zeros(arr_size)
        dBzdx = cp.zeros(arr_size)
        dBzdy = cp.zeros(arr_size)
        dBzdz = cp.zeros(arr_size)
        B = cp.zeros(arr_size)
        dBdx = cp.zeros(arr_size)
        dBdy = cp.zeros(arr_size)
        dBdz = cp.zeros(arr_size)

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
            B0,
            r_inner,
            t_axis,
            x_axis,
            y_axis,
            z_axis,
            Bxvec,
            Byvec,
            Bzvec,
            Exvec,
            Eyvec,
            Ezvec,
            Bx,
            By,
            Bz,
            Ex,
            Ey,
            Ez,
            dBxdx,
            dBxdy,
            dBxdz,
            dBydx,
            dBydy,
            dBydz,
            dBzdx,
            dBzdy,
            dBzdz,
            B,
            dBdx,
            dBdy,
            dBdz,
        )
        
        # need to account for dimensionalization of magnitude
        if self.negative_charge:
            B *= -1
            dBdx *= -1
            dBdy *= -1
            dBdz *= -1
        
        # Return values as tuple
        return _MultiInterpResult(
            Bx=Bx,
            By=By,
            Bz=Bz,
            Ex=Ex,
            Ey=Ey,
            Ez=Ez,
            dBxdx=dBxdx,
            dBxdy=dBxdy,
            dBxdz=dBxdz,
            dBydx=dBydx,
            dBydy=dBydy,
            dBydz=dBydz,
            dBzdx=dBzdx,
            dBzdy=dBzdy,
            dBzdz=dBzdz,
            B=B,
            dBdx=dBdx,
            dBdy=dBdy,
            dBdz=dBdz,
        )


@dataclass
class _MultiInterpResult:
    """Data container for return value of
    _DimensionalizedFieldModel.multi_interp()
    """
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
    
