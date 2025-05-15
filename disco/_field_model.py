"""Module for just the FieldModel class, to prevent circular imports."""

from astropy import units
import cupy as cp

from disco.kernels import multi_interp_kernel
from disco.constants import BLOCK_SIZE

class FieldModel:
    """Magnetic and electric field models used to propagate particles."""

    DEFAULT_RAW_B0 = 31e3 * units.nT

    def __init__(self, Bx, By, Bz, Ex, Ey, Ez, mass, charge, axes, B0=DEFAULT_RAW_B0):
        """Get an instance that is dimensionalized and stored on the GPU.

        mass is not part of field model, but is used to redimensionalize.
        Input argument should have astropy units attached.

        Parameters
        ----------
        Bx: array of shape (nx, ny, nz, nt), with units
          Magnetic Field X component
        By: array of shape (nx, ny, nz, nt), with units
          Magnetic Field Y component
        Bz: array of shape (nx, ny, nz, nt), with units
          Magnetic Field Z component
        Ex: array of shape (nx, ny, nz, nt), with units
          Electric Field X component
        Ey: array of shape (nx, ny, nz, nt), with units
          Electric Field Y component
        Ez: array of shape (nx, ny, nz, nt), with units
          Electric Field Z component
        """
        q = charge
        Re = constants.R_earth
        c = constants.c
        sf = q * Re / (mass * c**2)
        B_units = units.s / Re

        self.negative_charge = q.value < 0
        self.Bx = cp.array((sf * Bx).to(B_units).value)
        self.By = cp.array((sf * By).to(B_units).value)
        self.Bz = cp.array((sf * Bz).to(B_units).value)
        self.B0 = float((sf * B0).to(B_units).value)
        self.Ex = cp.array((sf * Ex).to(1).value)
        self.Ey = cp.array((sf * Ey).to(1).value)
        self.Ez = cp.array((sf * Ez).to(1).value)
        self.axes = axes

    def multi_interp(self, t, y, stopped_cutoff):
        """Interpolate field values at given positions.

        Paramaters
        ----------
        t: cupy array
          Vector of dimensionalized particle times
        y: cupy array
           Vector of shape (npart, 5) of particle states

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

        b0 = cp.zeros(arr_size) + self.B0
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

        bxvec = self.Bx.reshape(nttl, order="F")
        byvec = self.By.reshape(nttl, order="F")
        bzvec = self.Bz.reshape(nttl, order="F")
        exvec = self.Ex.reshape(nttl, order="F")
        eyvec = self.Ey.reshape(nttl, order="F")
        ezvec = self.Ez.reshape(nttl, order="F")

        bx = cp.zeros(arr_size)
        by = cp.zeros(arr_size)
        bz = cp.zeros(arr_size)
        ex = cp.zeros(arr_size)
        ey = cp.zeros(arr_size)
        ez = cp.zeros(arr_size)
        dbxdx = cp.zeros(arr_size)
        dbxdy = cp.zeros(arr_size)
        dbxdz = cp.zeros(arr_size)
        dbydx = cp.zeros(arr_size)
        dbydy = cp.zeros(arr_size)
        dbydz = cp.zeros(arr_size)
        dbzdx = cp.zeros(arr_size)
        dbzdy = cp.zeros(arr_size)
        dbzdz = cp.zeros(arr_size)
        b = cp.zeros(arr_size)
        dbdx = cp.zeros(arr_size)
        dbdy = cp.zeros(arr_size)
        dbdz = cp.zeros(arr_size)

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
            b0,
            r_inner,
            t_axis,
            x_axis,
            y_axis,
            z_axis,
            bxvec,
            byvec,
            bzvec,
            exvec,
            eyvec,
            ezvec,
            bx,
            by,
            bz,
            ex,
            ey,
            ez,
            dbxdx,
            dbxdy,
            dbxdz,
            dbydx,
            dbydy,
            dbydz,
            dbzdx,
            dbzdy,
            dbzdz,
            b,
            dbdx,
            dbdy,
            dbdz,
        )

        # Return values as tuple
        return (
            bx,
            by,
            bz,
            ex,
            ey,
            ez,
            dbxdx,
            dbxdy,
            dbxdz,
            dbydx,
            dbydy,
            dbydz,
            dbzdx,
            dbzdy,
            dbzdz,
            b,
            dbdx,
            dbdy,
            dbdz,
        )
