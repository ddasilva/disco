from dataclasses import dataclass
import math
import sys
from typing import Optional, List, Callable

from astropy.units import Quantity
import cupy as cp
import numpy as np

from disco._axes import Axes
from disco._dimensionalization import (
    dim_time,
    undim_time,
    dim_space,
    dim_momentum,
    dim_magnetic_moment,
    undim_magnetic_field,
    undim_space,
    undim_energy,
    undim_momentum,
    undim_magnetic_moment,
)
from disco._field_model import FieldModel
from disco._field_model_loader import FieldModelLoader, LazyFieldModelLoader, StaticFieldModelLoader
from disco._particle_history import ParticleHistory
from disco._kernels import do_step_kernel, rhs_kernel
from disco.constants import BLOCK_SIZE, NSTATE, RK45Coeffs

from astropy import units


__all__ = [
    "Axes",
    "FieldModel",
    "LazyFieldModelLoader",
    "ParticleHistory",
    "ParticleState",
    "TraceConfig",
    "trace_trajectory",
]


class TraceConfig:
    """Configuration for running the tracing code."""

    def __init__(
        self,
        t_final,
        output_freq=None,
        stopping_conditions=None,
        t_initial=0 * units.s,
        h_initial=1 * units.ms,
        rtol=1e-2,
        integrate_backwards=False,
        iters_max=None,
        reorder_freq=25,
    ):
        """Initialize a `TraceConfig` instance.

        Parameters
        ----------
        t_initial: scalar with time units
          Start time of integration
        t_final: scalar with time units
          end time of integration (set to inf seconds if you want to stop
          purely based on stopping conditions)
        output_freq: int or None
          How frequently (in iterations) to store output. Setting this
          to non-None means memory will accumulate with time.
        stopping_conditions: list of callables
          List of callables (functions) that return bools. Arguments are
          y, t, and field_model.
        h_initial: scalar with time units
          Initial step size in time (leave as positive even if integrating
          backwards)
        rtol: float
          Relative tolerance for adaptive integration
        integrate_backwards: bool
          Set to True to integrate backwards in time

        Examples
        --------
        Integrate between 0 and 10 seconds.

        >>> config = disco.TraceConfig(t_final=10 * units.s)

        Integrate backwards between 0 and -30 seconds.

        >>> from astropy import units
        >>> config = disco.TraceConfig(
              t_final=-30 * units.s,
              integrate_backwards=True
            )
        """
        self.t_final = t_final.to(units.s)
        self.output_freq = output_freq
        self.stopping_conditions = stopping_conditions or []
        self.t_initial = t_initial.to(units.s)
        self.h_initial = h_initial.to(units.ms)
        self.rtol = rtol
        self.integrate_backwards = integrate_backwards
        self.iters_max = iters_max
        self.reorder_freq = reorder_freq


class ParticleState:
    """Initial conditions of particles."""

    def __init__(self, x, y, z, ppar, magnetic_moment, mass, charge):
        """Initialize a `ParticleState` instance that is dimensionalized
        and stored on the GPU.

        All inputs are arrays except for mass and charge, which are
        single values.

        Parameters
        ----------
        x: array with units
          Starting X coordinate of particles
        y: array with units
          Starting Y coordinate of particles
        z: array with units
          Starting Z coordinate of particles
        ppar: array with units
          Starting parallel momentum of oparticles
        magnetic_moment: array with units
          First adiabatic invariant
        mass: scalar with units
          Mass of particles (all must be the same)
        charge: scalar with units
          Charge of particles (all must be the same)
        """
        self.x = cp.array(dim_space(x))
        self.y = cp.array(dim_space(y))
        self.z = cp.array(dim_space(z))
        self.ppar = cp.array(dim_momentum(ppar, mass))
        self.magnetic_moment = cp.array(dim_magnetic_moment(magnetic_moment, charge))

        self.mass = mass.to(units.kg)
        self.charge = charge.to(units.coulomb)


def trace_trajectory(config, particle_state, field_model, verbose=1):
    """Calculate a particle trajectory.

    Parameters
    ----------
    config : `disco.TraceConfig`
       Configuration for performing the trace
    particle_state : `disco.ParticleState`
       Initial conditions of the particles
    field_model : `disco.FieldModel` or `disco.FieldModelLoader`
       Magnetic and Electric field context
    verbose : int
      Set to zero to supress print statements

    Returns
    -------
    History of the trajectories as a `ParticleHistory` object. If `config.output_freq` is `None`, contains
    only the first and last step.

    Examples
    --------
    >>> history = disco.trace_trajectory(
          config, particle_state, field_model
        )
    >>> history.save("trajectories.h5")
    """
    # If passing a field model, make a static field model loader
    if isinstance(field_model, FieldModel):
        field_model = field_model.dimensionalize(particle_state.mass, particle_state.charge)
        field_model_loader = StaticFieldModelLoader(field_model)
    elif isinstance(field_model, FieldModelLoader):
        field_model_loader = field_model
    else:
        raise TypeError("field_model argument must be FieldModel or FieldModelLoader")

    # This implements the RK45 adaptive integration algorithm, with
    # absolute/relative tolerance and minimum/maximum step sizes
    npart = particle_state.x.size
    t_initial = dim_time(config.t_initial)
    t = cp.zeros(npart) + t_initial

    y = cp.zeros((npart, NSTATE))
    y[:, 0] = particle_state.x
    y[:, 1] = particle_state.y
    y[:, 2] = particle_state.z
    y[:, 3] = particle_state.ppar
    y[:, 4] = particle_state.magnetic_moment

    h = cp.zeros(npart) + dim_time(config.h_initial)
    if config.integrate_backwards:
        h *= -1

    t_final = dim_time(config.t_final)
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
    hist_stopped = []

    # Iteration
    # ---------------------------
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
            if verbose > 0:
                print("Reordering to reduce GPU load...")
            cur_reorder = cp.argsort(stopped)
            y = y[cur_reorder]
            t = t[cur_reorder]
            h = h[cur_reorder]
            stopped = stopped[cur_reorder]
            total_reorder = total_reorder[cur_reorder]
            stopped_cutoff = int(cp.searchsorted(stopped, cp.ones(1))[0])

        # Cupy broadcasting workaround (implicit broading doesn't work)
        h_ = cp.zeros((h.size, NSTATE))

        for i in range(NSTATE):
            h_[:, i] = h

        # Call _rhs() function to implement multiple function evaluations of
        # right hand side.
        k1, B, paused1 = _rhs(t, y, field_model_loader, stopped_cutoff)
        B = B.copy()
        k2, _, paused2 = _rhs(t + h * R.a2, y + h_ * R.b21 * k1, field_model_loader, stopped_cutoff)
        k3, _, paused3 = _rhs(
            t + h * R.a3, y + h_ * (R.b31 * k1 + R.b32 * k2), field_model_loader, stopped_cutoff
        )
        k4, _, paused4 = _rhs(
            t + h * R.a4,
            y + h_ * (R.b41 * k1 + R.b42 * k2 + R.b43 * k3),
            field_model_loader,
            stopped_cutoff,
        )
        k5, _, paused5 = _rhs(
            t + h * R.a5,
            y + h_ * (R.b51 * k1 + R.b52 * k2 + R.b53 * k3 + R.b54 * k4),
            field_model_loader,
            stopped_cutoff,
        )
        k6, _, paused6 = _rhs(
            t + h * R.a6,
            y + h_ * (R.b61 * k1 + R.b62 * k2 + R.b63 * k3 + R.b64 * k4 + R.b65 * k5),
            field_model_loader,
            stopped_cutoff,
        )

        k1 *= h_
        k2 *= h_
        k3 *= h_
        k4 *= h_
        k5 *= h_
        k6 *= h_
        paused = paused1 | paused2 | paused3 | paused4 | paused5 | paused6

        # Save incremented particles to history
        save_history = iter_count == 0 or (
            config.output_freq and (iter_count % config.output_freq == 0)
        )

        if save_history:
            total_reorder_rev = np.argsort(total_reorder)
            gamma, W = _calc_gamma_W(B, y)
            hist_t.append(t[total_reorder_rev].get())
            hist_y.append(y[total_reorder_rev].get())
            hist_B.append(B[total_reorder_rev].get())
            hist_W.append(W[total_reorder_rev].get())
            hist_h.append(h[total_reorder_rev].get())
            hist_stopped.append(stopped[total_reorder_rev].get())

        # Do runge-kutta step, check to change stopping state, and change step size
        # if step is performed
        num_iterated = _do_step(
            k1,
            k2,
            k3,
            k4,
            k5,
            k6,
            y,
            h,
            t,
            t_final,
            field_model_loader,
            stopped,
            paused,
            config,
            stopped_cutoff,
        )

        all_complete = cp.all(stopped)
        iter_count += 1

        # Print message to console if verbose enabled
        if verbose > 0:
            r_mean = cp.sqrt(y[:, 0] ** 2 + y[:, 1] ** 2 + y[:, 2] ** 2).mean()
            h_step = undim_time(float(h.mean())).to(units.ms).value

            t_progress = min((t.min() - t_initial) / (t_final - t_initial), 1)

            print(
                f"Time Complete: {100 * t_progress:.3f}% "
                f"Stopped: {100 * stopped.sum() / stopped.size:.3f}% "
                f"(iter {iter_count}, {num_iterated} iterated, h mean "
                f"{h_step:.2f} ms, r mean {r_mean:.2f})"
            )

            sys.stdout.flush()

    if verbose > 0:
        print(f"Took {iter_count} iterations")

    # We need to reverse of the accumulated reordering
    total_reorder_rev = np.argsort(total_reorder)

    assert cp.all(total_reorder[total_reorder_rev] == cp.arange(npart, dtype=int))

    # Always save last step of each, even if not recording full history
    gamma, W = _calc_gamma_W(B, y)
    hist_t.append(t[total_reorder_rev].get())
    hist_y.append(y[total_reorder_rev].get())
    hist_B.append(B[total_reorder_rev].get())
    hist_W.append(W[total_reorder_rev].get())
    hist_h.append(h[total_reorder_rev].get())
    hist_stopped.append(stopped[total_reorder_rev].get())

    # Prepare history object and return instance of ParticleHistory
    hist_t = undim_time(np.array(hist_t))
    hist_B = undim_magnetic_field(np.array(hist_B), particle_state.mass, particle_state.charge)
    hist_W = undim_energy(np.array(hist_W), particle_state.mass)
    hist_h = undim_time(np.array(hist_h))
    hist_stopped = np.array(hist_stopped)
    hist_y = np.array(hist_y)
    hist_pos_x = undim_space(hist_y[:, :, 0])
    hist_pos_y = undim_space(hist_y[:, :, 1])
    hist_pos_z = undim_space(hist_y[:, :, 2])
    hist_ppar = undim_momentum(hist_y[:, :, 3], particle_state.mass)
    hist_M = undim_magnetic_moment(hist_y[:, :, 4], particle_state.charge)

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
    )


def _rhs(t, y, field_model_loader, stopped_cutoff):
    """RIght hand side of the guiding center equation differential equation.

    Parameters
    -----------
    t: cupy array (nparticles,)
      Dimensionalized time variable
    y: cupy array (nparticles,)
      ODE state variable
    field_model_loader: FieldModelLoader
      Used to E and B fields and particles
    config: TraceConfig
      Tracing configuration

    Returns
    -------
    dydt: cupy array (nparticles, NSTATE)
      First three columns are position, fourth is parallel momentum,
      fifth is relativistic magnetic moment. This variable is
      dimensionalized. Filled up to `stopped_cutoff`.
    B: cupy array (nparticles,)
      Value of B at the particle positions, filled up to
      `stopped_cutoff`.
    """
    # Get B, E Values and partials
    interp_result, paused = field_model_loader.multi_interp(t, y, stopped_cutoff)

    # Launch Kernel to handle rest of RHS
    arr_size = y.shape[0]
    grid_size = int(math.ceil(stopped_cutoff / BLOCK_SIZE))

    r_inner = cp.zeros(arr_size) + field_model_loader.axes.r_inner
    dydt = cp.zeros((arr_size, NSTATE))

    rhs_kernel[grid_size, BLOCK_SIZE](
        dydt,
        y[:stopped_cutoff],
        t,
        paused,
        interp_result.Bx,
        interp_result.By,
        interp_result.Bz,
        interp_result.B,
        interp_result.Ex,
        interp_result.Ey,
        interp_result.Ez,
        interp_result.dBdx,
        interp_result.dBdy,
        interp_result.dBdz,
        interp_result.dBxdy,
        interp_result.dBxdz,
        interp_result.dBydx,
        interp_result.dBydz,
        interp_result.dBzdx,
        interp_result.dBzdy,
        field_model_loader.axes.x,
        field_model_loader.axes.y,
        field_model_loader.axes.z,
        field_model_loader.axes.t,
        field_model_loader.axes.x.size,
        field_model_loader.axes.y.size,
        field_model_loader.axes.z.size,
        field_model_loader.axes.t.size,
        r_inner,
    )

    return dydt, interp_result.B, paused


def _do_step(
    k1,
    k2,
    k3,
    k4,
    k5,
    k6,
    y,
    h,
    t,
    t_final,
    field_model_loader,
    stopped,
    paused,
    config,
    stopped_cutoff,
):
    """Do a Runge-Kutta Step.

    Paramters
    ---------
      k1-k6: K values for Runge-Kutta
      y: current state vector
      h: current vector of step sizes
      t: current vector of particle times
      t_final: final time (dimensionalized)
      stopped: boolean array of whether integration has fully stopped
      paused: boolean array of whether interpolation was skipped due to delayed loadeding
      field_model: instance of FieldModelLoader
      stopped: boolean array of whether integration has stopped
    Returns
    --------
    num_iterated: int
      Number of particles iterated
    """
    # Evaluate Stopping Conditions
    if config.stopping_conditions:
        for stop_cond in config.stopping_conditions:
            stopped |= stop_cond(y, t, field_model_loader)

    # Nan signals some major problem in the code, better to stop immediately
    stopped |= cp.isnan(h)

    # Call Kernel to do the rest of the work
    grid_size = int(math.ceil(stopped_cutoff / BLOCK_SIZE))
    arr_size = y.shape[0]
    y_next = cp.zeros(y.shape)
    z_next = cp.zeros(y.shape)
    rtol_arr = cp.zeros(arr_size) + config.rtol
    t_final_arr = cp.zeros(arr_size) + t_final
    r_inner = cp.zeros(arr_size) + field_model_loader.axes.r_inner
    mask = cp.zeros(arr_size, dtype=bool)

    do_step_kernel[grid_size, BLOCK_SIZE](
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
        paused,
        field_model_loader.axes.x,
        field_model_loader.axes.x.size,
        field_model_loader.axes.y,
        field_model_loader.axes.y.size,
        field_model_loader.axes.z,
        field_model_loader.axes.z.size,
        field_model_loader.axes.t,
        field_model_loader.axes.t.size,
        r_inner,
        config.integrate_backwards,
    )

    num_iterated = mask.sum()

    return num_iterated


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
