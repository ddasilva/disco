from dataclasses import dataclass
import math
import sys
from typing import Any, Optional, List, Callable

from astropy.units import Quantity
import cupy as cp
import numpy as np

from disco.constants import RK45Coeffs
from disco.kernels import (
    do_step_kernel,
    multi_interp_kernel,
    rhs_kernel,
)

from astropy import constants, units


@dataclass
class TraceConfig:
    """Configuration for running the tracing code.

    Attributes
    ----------
    t_final: Quantity (time)
      end time of integration (set to inf seconds if you want to stop
      purely based on stopping conditions)
    output_freq: int or None
      How frequently (in iterations) to store output. Setting this
      to non-None means memory will accumulate with time.
    stopping_conditions: list of callables
      List of callables (functions) that return bools. Arguments are
      y, t, and field_model.
    t_initial: Quantity (time)
      Start time of integration
    h_initial: Quantity (time)
      Initial step size in time (leave as positive even if integrating
      backwards)
    rtol: float
      Relative tolerance for adaptive integration
    integrate_backwards: bool
      Set to True to integrate backwards in time
    """

    t_final: Quantity
    output_freq: Optional[int] = None
    stopping_conditions: Optional[List[Callable]] = None
    t_initial: Quantity = 0 * units.s
    h_initial: Quantity = 1 * units.ms
    rtol: float = 1e-2
    integrate_backwards: bool = False
    iters_max: Optional[int] = None
    reorder_freq: int = 25


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
        block_size = 256
        grid_size = int(math.ceil(stopped_cutoff / block_size))

        multi_interp_kernel[grid_size, block_size](
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


class ParticleState:
    """1D arrays of cartesian particle position component"

    Attributes
    ----------
    x: x position
    y: y position
    z: z position
    ppar: parallel momentum
    magnetic_moment: first invariant
    mass: rest mass
    """

    def __init__(self, x, y, z, ppar, magnetic_moment, mass, charge):
        """Get a ParticleState() instance that is dimensionalized
        and stored on the GPU.

        Input argument should have astropy units attached, except
        `charge'.
        Returns ParticleState instance
        """
        # Using redimensionalization of Elkington et al., 2002
        q = charge
        Re = constants.R_earth
        c = constants.c

        self.x = cp.array((x / Re).to(1).value)
        self.y = cp.array((y / Re).to(1).value)
        self.z = cp.array((z / Re).to(1).value)
        self.ppar = cp.array((ppar / (c * mass)).to(1).value)
        self.magnetic_moment = cp.array((magnetic_moment / (q * Re)).to(Re / units.s).value)
        self.mass = cp.array(mass.to(units.kg).value)


class Axes:
    """1D arrays of rectilinear grid axes

    Attributes
      x: x axis
      y: y axis
      z: z axis
      t: time axis
      r_inner: inner boundary
    """

    def __init__(self, x, y, z, t, r_inner):
        """Initialize instance that is dimensionalized and stored
        on the GPU.

        Input arguments should have astropy units
        """
        assert len(x.shape) == 1
        assert len(y.shape) == 1
        assert len(z.shape) == 1
        assert len(t.shape) == 1

        Re = constants.R_earth
        self.t = cp.array(_redim_time(t))
        self.x = cp.array((x / Re).to(1).value)
        self.y = cp.array((y / Re).to(1).value)
        self.z = cp.array((z / Re).to(1).value)
        self.r_inner = (r_inner / Re).to(1).value

    def get_neighbors(self, t, pos_x, pos_y, pos_z):
        """Get instance of RectilinearNeighbors specifying surrounding
        cell through indeces of upper corner

        Returns instance of _RectilinearNeighbors
        """
        field_i = cp.searchsorted(self.x, pos_x)
        field_j = cp.searchsorted(self.y, pos_y)
        field_k = cp.searchsorted(self.z, pos_z)
        field_l = cp.searchsorted(self.t, t, side="right")

        return _RectilinearNeighbors(
            field_i=field_i,
            field_j=field_j,
            field_k=field_k,
            field_l=field_l,
        )


@dataclass
class ParticleHistory:
    """History of positions, parallel momentum, and useful values.."""

    t: Any
    x: Any
    y: Any
    z: Any
    ppar: Any
    B: Any  # local field strength
    W: Any  # energy
    h: Any  # step size


@dataclass
class _RectilinearNeighbors:
    """Neighbors of given particles, used for interpolation"""

    field_i: Any
    field_j: Any
    field_k: Any
    field_l: Any


def trace_trajectory(config, particle_state, field_model, verbose=1):
    """Calculate a particle trajectory.

    Parameters
    ----------
    config: TraceConfig
       Configuration for perorming the trace
    particle_state: ParticleState
       Initial conditions of the particles
    field_model: FieldModel
       Magnetic and Electric field context
    verbose: int
      Set to zero to supress print statements

    Returns
    --------
    hist: ParticleHistory
       History of the trace. If output_freq=None, contains only the last
       step.
    """
    # This implements the RK45 adaptive integration algorithm, with
    # absolute/relative tolerance and minimum/maximum step sizes
    npart = particle_state.x.size
    nstate = 5
    t = cp.zeros(npart) + _redim_time(config.t_initial)

    y = cp.zeros((npart, nstate))
    y[:, 0] = particle_state.x
    y[:, 1] = particle_state.y
    y[:, 2] = particle_state.z
    y[:, 3] = particle_state.ppar
    y[:, 4] = particle_state.magnetic_moment

    h = cp.zeros(npart) + _redim_time(config.h_initial)
    if config.integrate_backwards:
        h *= -1

    t_final = _redim_time(config.t_final)
    all_complete = False
    stopped = cp.zeros(npart, dtype=bool)
    total_reorder = cp.arange(npart, dtype=int)
    stopped_cutoff = npart
    iter_count = 0

    R = RK45Coeffs

    hist_y = []
    hist_t = []
    hist_W = []
    hist_B = []
    hist_h = []

    while True:
        # Stop iterating when all_complete flag is set
        if all_complete:
            break

        # Stop iterating if exceeding the maximum iterations
        if config.iters_max and iter_count >= config.iters_max:
            break

        # Reorder batches based on stopped flag to group stopped particles
        # on same warp (avoid blocking)
        if config.reorder_freq is not None and (iter_count % config.reorder_freq == 0):
            print("Reordering to reduce GPU load...")
            cur_reorder = cp.argsort(stopped)
            y = y[cur_reorder]
            t = t[cur_reorder]
            h = h[cur_reorder]
            stopped = stopped[cur_reorder]
            total_reorder = total_reorder[cur_reorder]
            stopped_cutoff = int(cp.searchsorted(stopped, cp.ones(1))[0])
            # print("Stopped cutoff:", stopped_cutoff)

        # Cupy broadcasting workaround (implicit broading doesn't work)
        h_ = cp.zeros((h.size, nstate))

        for i in range(nstate):
            h_[:, i] = h

        # Call _rhs() function to implement multiple function evaluations of
        # right hand side.
        k1, B = _rhs(t, y, field_model, config, stopped_cutoff)
        k2, _ = _rhs(t + h * R.a2, y + h_ * R.b21 * k1, field_model, config, stopped_cutoff)
        k3, _ = _rhs(
            t + h * R.a3, y + h_ * (R.b31 * k1 + R.b32 * k2), field_model, config, stopped_cutoff
        )
        k4, _ = _rhs(
            t + h * R.a4,
            y + h_ * (R.b41 * k1 + R.b42 * k2 + R.b43 * k3),
            field_model,
            config,
            stopped_cutoff,
        )
        k5, _ = _rhs(
            t + h * R.a5,
            y + h_ * (R.b51 * k1 + R.b52 * k2 + R.b53 * k3 + R.b54 * k4),
            field_model,
            config,
            stopped_cutoff,
        )
        k6, _ = _rhs(
            t + h * R.a6,
            y + h_ * (R.b61 * k1 + R.b62 * k2 + R.b63 * k3 + R.b64 * k4 + R.b65 * k5),
            field_model,
            config,
            stopped_cutoff,
        )

        k1 *= h_
        k2 *= h_
        k3 *= h_
        k4 *= h_
        k5 *= h_
        k6 *= h_

        # Save incremented particles to history
        if config.output_freq is not None and (iter_count % config.output_freq == 0):
            total_reorder_rev = np.argsort(total_reorder)
            gamma = cp.sqrt(1 + 2 * B * y[:, 4] + y[:, 3] ** 2)
            W = gamma - 1
            hist_t.append(t[total_reorder_rev].get())
            hist_y.append(y[total_reorder_rev].get())
            hist_B.append(B[total_reorder_rev].get())
            hist_W.append(W[total_reorder_rev].get())
            hist_h.append(h[total_reorder_rev].get())

        # Do runge-kutta step, check to change stopping state, and change step size
        # if step is performed
        num_iterated = _do_step(
            k1, k2, k3, k4, k5, k6, y, h, t, t_final, field_model, stopped, config, stopped_cutoff
        )

        all_complete = cp.all(stopped)
        iter_count += 1

        # Print message to console if verbose enabled
        if verbose > 0:
            r_mean = cp.sqrt(y[:, 0] ** 2 + y[:, 1] ** 2 + y[:, 2] ** 2).mean()
            h_step = _undim_time(float(h.mean())).to(units.ms).value

            print(
                f"Time Complete: {100 * min(t.min() / t_final, 1):.3f}% "
                f"Stopped: {100 * stopped.sum() / stopped.size:.3f}% "
                f"(iter {iter_count}, {num_iterated} iterated, h mean "
                f"{h_step:.2f} ms, r mean {r_mean:.2f})"
            )

            sys.stdout.flush()

    if verbose > 0:
        print(f"Took {iter_count} iterations")

    # We need to reverse of the accumulated reordering
    total_reorder_rev = np.argsort(total_reorder)

    # Always save last step of each, even if not recording full history
    gamma = cp.sqrt(1 + 2 * B * y[:, 4] + y[:, 3] ** 2)
    W = gamma - 1
    hist_t.append(t[total_reorder_rev].get())
    hist_y.append(y[total_reorder_rev].get())
    hist_B.append(B[total_reorder_rev].get())
    hist_W.append(W[total_reorder_rev].get())
    hist_h.append(h[total_reorder_rev].get())

    # Prepare history object and return instance of ParticleHistory
    hist_t = np.array(hist_t)
    hist_B = np.array(hist_B)
    hist_W = np.array(hist_W)
    hist_h = np.array(hist_h)
    hist_y = np.array(hist_y)
    hist_pos_x = hist_y[:, :, 0]
    hist_pos_y = hist_y[:, :, 1]
    hist_pos_z = hist_y[:, :, 2]
    hist_ppar = hist_y[:, :, 3]

    return ParticleHistory(
        t=hist_t,
        x=hist_pos_x,
        y=hist_pos_y,
        z=hist_pos_z,
        ppar=hist_ppar,
        B=hist_B,
        W=hist_W,
        h=hist_h,
    )


def _rhs(t, y, field_model, config, stopped_cutoff):
    """RIght hand side of the guiding center equation differential equation.

    Args
      t: ODE derivative variable
      y: ODE state variable
      field_model: instance of FieldModel (provides E and B fields)
      axes: instance of Axes (rectilinear grid axes)
      config: instance of Config (tracing configuration)
    Returns
      dydt: cupy array (nparticles, 5). First three columns are position,
        fourth is parallel momentum, fifth is relativistic magnetic moment
    """
    # Get B Values
    (
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
    ) = field_model.multi_interp(t, y, stopped_cutoff)

    # need to account for dimensionalization of magnitude
    if field_model.negative_charge:
        B *= -1
        dBdx *= -1
        dBdy *= -1
        dBdz *= -1

    # Launch Kernel to handle rest of RHS
    arr_size = y.shape[0]
    block_size = 256
    grid_size = int(math.ceil(stopped_cutoff / block_size))

    r_inner = cp.zeros(arr_size) + field_model.axes.r_inner
    dydt = cp.zeros((arr_size, 5))

    rhs_kernel[grid_size, block_size](
        y[:stopped_cutoff],
        t,
        Bx,
        By,
        Bz,
        B,
        Ex,
        Ey,
        Ez,
        field_model.axes.x,
        field_model.axes.y,
        field_model.axes.z,
        field_model.axes.t,
        field_model.axes.x.size,
        field_model.axes.y.size,
        field_model.axes.z.size,
        field_model.axes.t.size,
        r_inner,
        dBdx,
        dBdy,
        dBdz,
        dBxdy,
        dBxdz,
        dBydx,
        dBydz,
        dBzdx,
        dBzdy,
        dydt,
    )

    return dydt, B


def _do_step(
    k1, k2, k3, k4, k5, k6, y, h, t, t_final, field_model, stopped, config, stopped_cutoff
):
    """Do a Runge-Kutta Step.

    Args
      k1-k6: K values for Runge-Kutta
      y: current state vector
      h: current vector of step sizes
      t: current vector of particle times
      t_final: final time (dimensionalized)
      field_model: instance of libgputrace.FieldModel
      stopped: boolean array of whether integration has stopped
    Returns
      num_iterated: number of particles iterated
    """
    # Evaluate Stopping Conditions
    if config.stopping_conditions:
        for stop_cond in config.stopping_conditions:
            stopped |= stop_cond(y, t, field_model)

    # Nan signals some major problem in the code, better to stop immediately
    stopped |= cp.isnan(h)

    # Call Kernel to do the rest of the work
    arr_size = y.shape[0]
    block_size = 1024
    grid_size = int(math.ceil(stopped_cutoff / block_size))
    y_next = cp.zeros(y.shape)
    z_next = cp.zeros(y.shape)
    rtol_arr = cp.zeros(arr_size) + config.rtol
    t_final_arr = cp.zeros(arr_size) + t_final
    r_inner = cp.zeros(arr_size) + field_model.axes.r_inner
    mask = cp.zeros(arr_size, dtype=bool)

    do_step_kernel[grid_size, block_size](
        k1,
        k2,
        k3,
        k4,
        k5,
        k6,
        y[:stopped_cutoff],
        y_next,
        z_next,
        h,
        t,
        rtol_arr,
        t_final_arr,
        mask,
        stopped,
        field_model.axes.x,
        field_model.axes.x.size,
        field_model.axes.y,
        field_model.axes.y.size,
        field_model.axes.z,
        field_model.axes.z.size,
        field_model.axes.t,
        field_model.axes.t.size,
        r_inner,
        config.integrate_backwards,
    )

    num_iterated = mask.sum()

    return num_iterated


def _redim_time(val):
    """Redimensionalize a time value.

    Args
      value: value with units
    Returns
      value in redimensionalized units
    """
    sf = constants.c / constants.R_earth
    return (sf * val).to(1).value


def _undim_time(val):
    """Redimensionalize a time value.

    Args
      value: value in seconds
    Returns
      value in redimensionalized units
    """
    sf = constants.R_earth / constants.c
    return val * sf
