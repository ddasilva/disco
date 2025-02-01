import cupy as cp
import sys
from cupyx import jit
import math
from dataclasses import dataclass
from typing import Any


@dataclass
class TraceConfig:
    """Configuration for running the tracing code"""
    max_iters: int            # number of iterations
    dt: float                 # units: seconds
    grad_step: float = 5e-2   # finite diff delta step (half)

    
@dataclass
class ParticleState:
    """1D arrays of cartesian particle position component"""
    # these vary in time
    x: Any                    # units: earth radii
    y: Any                    # units: earth radii
    z: Any                    # units: earth radii
    vpar: Any                 # units: earth radii / s

    # these *don't* vary in time
    magnetic_moment: Any      # first invariant, units: 1e9 (R_earth**2) A
    mass: Any                 # units: kilograms
    charge: Any               # units: columbs

    
@dataclass
class Axes:
    """1D arrays of rectilinear grid axes"""
    x: Any                    # units: earth radii
    y: Any                    # units: earth radii
    z: Any                    # units: earth radii
    
    
@dataclass
class ParticleHistory:
    """History of positions, velocity, and ambient field 
    strength (to calculate vperp).
    """
    def __init__(self, size, num_particles):
        self.x = cp.zeros((size, num_particles))
        self.y = cp.zeros((size, num_particles))
        self.z = cp.zeros((size, num_particles))
        self.vpar = cp.zeros((size, num_particles))
        self.Btot = cp.zeros((size, num_particles))

        
@dataclass
class Neighbors:
    """Neighbors of given particles, used for interpolation"""
    field_i: Any
    field_j: Any
    field_k: Any


def trace_trajectory(config, particle_state, hist, Bx, By, Bz, Btot, Ex, Ey, Ez, axes):
    """Perform a euler integration particle trace.
    
    Works on a rectilinear grid. 7

    Args
      config: instance of libgputrace.TraceConfig
      particle_state: instance of libgputrace.ParticleState
      hist: instance of libgputrace.ParticleHistory
      Bx, By, Bz: 3D magnetic field in units of nT
      Ex, Ey, Ez: 3D magnetic field in TBD units
      axes: instance of libgputrace.Axes
    """
    for i in range(config.max_iters):
        dydt = rhs(particle_state, Bx, By, Bz, Btot, Ex, Ey, Ez, axes, config)

        particle_state.x += config.dt * dydt[:, 0]
        particle_state.y += config.dt * dydt[:, 1]
        particle_state.z += config.dt * dydt[:, 2]
        particle_state.vpar += config.dt * dydt[:, 3]

        # Record history of positions
        hist.x[i, :] = particle_state.x
        hist.y[i, :] = particle_state.y
        hist.z[i, :] = particle_state.z
        hist.vpar[i, :] = particle_state.vpar

        print('.', end='')
        sys.stdout.flush()

    print()


def rhs(particle_state, Bx, By, Bz, Btot, Ex, Ey, Ez, axes, config):
    # Get B and Bhat vectors at particle locations
    Bx_cur, neighbors = interp_field(Bx, particle_state, axes)
    By_cur, _ = interp_field(By, particle_state, axes, neighbors=neighbors)
    Bz_cur, _ = interp_field(Bz, particle_state, axes, neighbors=neighbors)
    Btot_cur, _ = interp_field(Btot, particle_state, axes, neighbors=neighbors)
    Ex_cur, _ = interp_field(Ex, particle_state, axes, neighbors=neighbors)
    Ey_cur, _ = interp_field(Ey, particle_state, axes, neighbors=neighbors)
    Ez_cur, _ = interp_field(Ez, particle_state, axes, neighbors=neighbors)

    Bx_hat_cur = Bx_cur / Btot_cur
    By_hat_cur = By_cur / Btot_cur
    Bz_hat_cur = Bz_cur / Btot_cur
            
    # Gradient B drift and curvature drift guiding center term
    gradB_x = (
        interp_field(Btot, particle_state, axes, dx=config.grad_step)[0]
        - interp_field(Btot, particle_state, axes, dx=-config.grad_step)[0]
    ) / (2 * config.grad_step)
    gradB_y = (
        interp_field(Btot, particle_state, axes, dy=config.grad_step)[0]
        - interp_field(Btot, particle_state, axes, dy=-config.grad_step)[0]
    ) / (2 * config.grad_step)
    gradB_z = (
    interp_field(Btot, particle_state, axes, dz=config.grad_step)[0]
        - interp_field(Btot, particle_state, axes, dz=-config.grad_step)[0]
    ) / (2 * config.grad_step)
    
    cross_term1 = cp.zeros((Bx_cur.size, 3))
    cross_term1[:, 0] = Bx_cur
    cross_term1[:, 1] = By_cur
    cross_term1[:, 2] = Bz_cur
    
    cross_term2 = cp.zeros((Bx_cur.size, 3))
    cross_term2[:, 0] = gradB_x
    cross_term2[:, 1] = gradB_y
    cross_term2[:, 2] = gradB_z 
    
    v_B = cp.zeros((Bx_cur.size, 3))
    v_B[:, :] = cp.cross(cross_term1, cross_term2)
    
    v_perp = cp.sqrt(            # unit constant cancels 
        particle_state.magnetic_moment * Btot_cur / particle_state.mass
    )
    
    scale_term = particle_state.mass * (
        (v_perp**2 + 2 * particle_state.vpar**2)
        / (2 * particle_state.charge * Btot_cur**3)
    )
    #scale_term *= 1e9
    scale_term *= 1e13
    
    for i in range(3):
        v_B[:, i] *= scale_term
        
    # Calculate ExB drift
    cross_term2 = cp.zeros((Ex_cur.size, 3))
    cross_term2[:, 0] = Ex_cur
    cross_term2[:, 1] = Ey_cur
    cross_term2[:, 2] = Ez_cur
    
    v_ExB = cp.zeros((Ex_cur.size, 3))
    v_ExB[:, :] = cp.cross(cross_term1, cross_term2)
    scale_term = 1 / Btot_cur**2
    
    for i in range(3):
        v_ExB[:, i] *= scale_term

    # bhat dot gradB
    grad_step_x = config.grad_step * Bx_hat_cur
    grad_step_y = config.grad_step * By_hat_cur 
    grad_step_z = config.grad_step * Bz_hat_cur
    
    bhat_dot_gradB = (
        interp_field(
            Btot, particle_state, axes, 
            dx=grad_step_x, dy=grad_step_y, dz=grad_step_z
        )[0]
        -
        interp_field(
            Btot, particle_state, axes, 
            dx=-grad_step_x, dy=-grad_step_y, dz=-grad_step_z
        )[0]         
    ) / (2 * config.grad_step)
    
    
    # Set the derivative
    dydt = cp.zeros((particle_state.x.size, 4))        
    dydt[:, 0] = particle_state.vpar * Bx_hat_cur + v_B[:, 0] + v_ExB[:, 0]
    dydt[:, 1] = particle_state.vpar * By_hat_cur + v_B[:, 1] + v_ExB[:, 1]
    dydt[:, 2] = particle_state.vpar * Bz_hat_cur + v_B[:, 2] + v_ExB[:, 2]
    dydt[:, 3] = - (
        (particle_state.magnetic_moment / particle_state.mass)
        * bhat_dot_gradB
    )
    
    return dydt


    
def interp_field(field, particle_state, axes, neighbors=None, dx=0, dy=0, dz=0):
    """Interpolate a 3D gridded field at given positions.

    Args
      field: 3D gridded field to interpolate
      partilce_state: instance of ParticleState
      axes: rectilinear grid axes
      neighbors: optional, reuse this value for lookup of the neighbors
      dx, dy, dz: optional, perturb the position by these values    
    Return
      result: cupy array of interpolated field values at position
      neighbors: neighbors object for reuse
    """ 
    pos_x_pert = particle_state.x + dx
    pos_y_pert = particle_state.y + dy
    pos_z_pert = particle_state.z + dz
    
    if neighbors is None:
        neighbors = Neighbors(
            field_i=cp.searchsorted(axes.x, pos_x_pert),
            field_j=cp.searchsorted(axes.y, pos_y_pert),
            field_k=cp.searchsorted(axes.z, pos_z_pert),
        )
        
    result = cp.zeros(particle_state.x.shape)    
    arr_size = particle_state.x.size
    block_size = 1024
    grid_size = int(math.ceil(arr_size / block_size))
    
    interp_field_kernel[grid_size, block_size](
        arr_size, result, field,
        neighbors.field_i, neighbors.field_j, neighbors.field_k,
        pos_x_pert, pos_y_pert, pos_z_pert,
        axes.x, axes.y, axes.z
    )

    return result, neighbors
    

@jit.rawkernel()
def interp_field_kernel(
        arr_size, result, field,
        field_i, field_j, field_k,
        pos_x, pos_y, pos_z,
        x_axis, y_axis, z_axis
):
    """[CUPY KERNEL] Interpolate field using neighbors.
    
    Uses inverse distance weighted average. This kernel operates
    on one position per thread.
    """
    idx = jit.blockDim.x * jit.blockIdx.x + jit.threadIdx.x
    
    if idx < arr_size:    
        w_accum = 0.0
    
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    dist_x = x_axis[field_i[idx] + di] - pos_x[idx]
                    dist_y = y_axis[field_j[idx] + dj] - pos_y[idx]
                    dist_z = z_axis[field_k[idx] + dk] - pos_z[idx]
                    
                    w = 1 / cp.sqrt(dist_x**2 + dist_y**2 + dist_z**2)                
                    f = field[field_i[idx] + di, field_j[idx] + dj, field_k[idx] + dk]
                    
                    result[idx] += w * f
                    w_accum += w
    
        result[idx] /= w_accum
