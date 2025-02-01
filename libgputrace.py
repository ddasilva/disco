import cupy as cp
import sys
from cupyx import jit
import math
from dataclasses import dataclass
from typing import Any
from astropy import constants, units
from scipy.constants import elementary_charge


@dataclass
class TraceConfig:
    """Configuration for running the tracing code"""
    max_iters: int            # number of iterations
    dt: float                 # units: seconds
    grad_step: float = 5e-2   # finite diff delta step (half)

    
@dataclass
class FieldModel:
    """Set of magnetic and electric field models

    See also:
      Axes
    """
    Bx: Any
    By: Any
    Bz: Any
    B: Any
    Ex: Any
    Ey: Any
    Ez: Any

    _dimensionalized: bool

    @classmethod
    def initialize(cls, Bx, By, Bz, B, Ex, Ey, Ez, mass):
        """Get a FieldModel() instance that is dimensionalized 
        and stored on the GPU.

        mass is not part of field model, but is used for 
        to redimensionalize.

        Input argument should have astropy units attached.
        Returns FieldModel instance         
        """
        q = elementary_charge * units.C
        Re = constants.R_earth
        c = constants.c
        sf = (q * Re / (mass * c**2))
        B_units = units.s / units.km
        
        gpu_Bx = cp.array((sf * Bx).to(B_units).value)
        gpu_By = cp.array((sf * By).to(B_units).value)
        gpu_Bz = cp.array((sf * Bz).to(B_units).value)
        gpu_B = cp.array((sf * B).to(B_units).value)        
        gpu_Ex = cp.array((sf * Ex).to(1).value)
        gpu_Ey = cp.array((sf * Ey).to(1).value)
        gpu_Ez = cp.array((sf * Ez).to(1).value)
        
        return FieldModel(
            Bx=gpu_Bx, By=gpu_By, Bz=gpu_Bz, B=gpu_B,
            Ex=gpu_Ex, Ey=gpu_Ey, Ez=gpu_Ez,
            _dimensionalized=True
            
        )


@dataclass
class ParticleState:
    """1D arrays of cartesian particle position component"""
    # these vary in time
    x: Any                    # x position
    y: Any                    # y position
    z: Any                    # z position
    vpar: Any                 # parallel velocity

    # these *don't* vary in time
    magnetic_moment: Any      # first invariant
    mass: Any                 # rest mass
    charge: Any               # charge

    _dimensionalized: bool

    @classmethod
    def initialize(cls, x, y, z, vpar, magnetic_moment, mass, charge):
        """Get a ParticleState() instance that is dimensionalized 
        and stored on the GPU.
        
        Input argument should have astropy units attached.
        Returns ParticleState instance         
        """
        # Using redimensionalization of Elkington et al., 2002
        q = elementary_charge * units.C
        Re = constants.R_earth
        c = constants.c
        
        gpu_x = cp.array((x / Re).to(1).value)
        gpu_y = cp.array((y / Re).to(1).value)
        gpu_z = cp.array((z / Re).to(1).value)
        gpu_vpar = cp.array((vpar / c).to(1).value)
        gpu_magnetic_moment = cp.array(
            (magnetic_moment / (q * Re)).to(units.km/units.s).value
        )
        gpu_mass = cp.array(mass.to(units.kg).value)
        gpu_charge = cp.array(charge.to(units.C).value)
        
        return ParticleState(
            x=gpu_x, y=gpu_y, z=gpu_z, vpar=gpu_vpar,
            magnetic_moment=gpu_magnetic_moment, mass=gpu_mass,
            charge=gpu_charge, _dimensionalized=True        
        )
        
        
@dataclass
class Axes:
    """1D arrays of rectilinear grid axes"""
    x: Any                    # x axis
    y: Any                    # y axis
    z: Any                    # z axis
    _dimensionalized: bool
    
    @classmethod
    def initialize(cls, x, y, z):
        """Get Axes() instance that is dimensionalized and stored
        on the GPU.

        INput argument should have astropy units
        Returns Axes instance
        """        
        Re = constants.R_earth
        gpu_x = cp.array((x / Re).to(1).value)
        gpu_y = cp.array((y / Re).to(1).value)
        gpu_z = cp.array((z / Re).to(1).value)

        return Axes(x=gpu_x, y=gpu_y, z=gpu_z, _dimensionalized=True)
        
    
@dataclass
class ParticleHistory:
    """History of positions, velocity, and ambient field 
    strength (to calculate vperp).
    """
    t: Any
    x: Any
    y: Any
    z: Any
    vpar: Any
    B: Any      # local field strength
    W: Any      # energy
    
    @classmethod
    def initialize(cls, size, num_particles):
        t = cp.zeros((size, num_particles))
        x = cp.zeros((size, num_particles))
        y = cp.zeros((size, num_particles))
        z = cp.zeros((size, num_particles))
        vpar = cp.zeros((size, num_particles))
        B = cp.zeros((size, num_particles))
        W = cp.zeros((size, num_particles))

        return ParticleHistory(t=t,  x=x, y=y, z=z, vpar=vpar, B=B, W=W)
        
        
@dataclass
class Neighbors:
    """Neighbors of given particles, used for interpolation"""
    field_i: Any
    field_j: Any
    field_k: Any


def trace_trajectory(config, particle_state, hist, field_model, axes):
    """Perform a euler integration particle trace.
    
    Works on a rectilinear grid.

    Args
      config: instance of libgputrace.TraceConfig
      particle_state: instance of libgputrace.ParticleState
      hist: instance of libgputrace.ParticleHistory
      field_model: instance of libgputrace.FieldModel
      axes: instance of libgputrace.Axes
    """
    if not axes._dimensionalized:
        raise ValueError(
            'axes must be created with Axes.initialize()'
        )        
    if not particle_state._dimensionalized:
        raise ValueError(
            'paritcle_state must be created with ParticleState.initialize()'
        )
    if not field_model._dimensionalized:
        raise ValueError(
            'field_model must be created with FieldModel.initialize()'
        )

    # dimensionalize timestep
    t = cp.zeros(particle_state.x.size)
    dt = ((constants.c/constants.R_earth) * config.dt * units.s).to(1).value

    for i in range(config.max_iters):
        dydt, B, W = rhs(particle_state, field_model, axes, config)

        t += dt
        particle_state.x += dt * dydt[:, 0]
        particle_state.y += dt * dydt[:, 1]
        particle_state.z += dt * dydt[:, 2]
        particle_state.vpar += dt * dydt[:, 3]

        # Record history of positions
        hist.x[i, :] = particle_state.x 
        hist.x[i, :] = particle_state.x
        hist.y[i, :] = particle_state.y
        hist.z[i, :] = particle_state.z
        hist.vpar[i, :] = particle_state.vpar
        hist.B[i, :] = B
        hist.W[i, :] = W
        hist.t[i, :] = t
        
        print('.', end='')
        sys.stdout.flush()

    print()


def rhs(particle_state, field_model, axes, config):
    """RIght hand side of the guiding center equation differential equation.

    Args
      particle_state: instance of ParticleState (holds particle info)
      field_model: instance of FieldModel (provides E and B fields)
      axes: instance of Axes (rectilinear grid axes)
      config: instance of Config (tracing configuration)
    Returns
      dydt: cupy array (nparticles, 4). First three columns are position, fourth
        is parallel velocityn
      B: redimensionalized magnetic field strength
      W: redimensionalized energy
    """
    # Get B and E at particleposition
    Bx, neighbors = interp_field(field_model.Bx, particle_state, axes)
    By, _ = interp_field(field_model.By, particle_state, axes, neighbors=neighbors)
    Bz, _ = interp_field(field_model.Bz, particle_state, axes, neighbors=neighbors)
    B, _ = interp_field(field_model.B, particle_state, axes, neighbors=neighbors)
    Ex, _ = interp_field(field_model.Ex, particle_state, axes, neighbors=neighbors)
    Ey, _ = interp_field(field_model.Ey, particle_state, axes, neighbors=neighbors)
    Ez, _ = interp_field(field_model.Ez, particle_state, axes, neighbors=neighbors)
    
    # Get derivatives from finite difference
    eps = config.grad_step
    
    dBdx = (
        interp_field(field_model.B, particle_state, axes, dx=eps)[0]
        - interp_field(field_model.B, particle_state, axes, dx=-eps)[0]
    ) / (2 * eps)
    dBdy = (
        interp_field(field_model.B, particle_state, axes, dy=eps)[0]
        - interp_field(field_model.B, particle_state, axes, dy=-eps)[0]
    ) / (2 * eps)
    dBdz = (
        interp_field(field_model.B, particle_state, axes, dz=eps)[0]
        - interp_field(field_model.B, particle_state, axes, dz=-eps)[0]
    ) / (2 * eps)
        
    # dBxdx = (
    #     interp_field(field_model.Bx, particle_state, axes, dx=eps)[0]
    #     - interp_field(field_model.Bx, particle_state, axes, dx=-eps)[0]
    # ) / (2 * eps)
    dBxdy = (
        interp_field(field_model.Bx, particle_state, axes, dy=eps)[0]
        - interp_field(field_model.Bx, particle_state, axes, dy=-eps)[0]
    ) / (2 * eps)
    dBxdz = (
        interp_field(field_model.Bx, particle_state, axes, dz=eps)[0]
        - interp_field(field_model.Bx, particle_state, axes, dz=-eps)[0]
    ) / (2 * eps)


    dBydx = (
        interp_field(field_model.By, particle_state, axes, dx=eps)[0]
        - interp_field(field_model.By, particle_state, axes, dx=-eps)[0]
    ) / (2 * eps)
    # dBydy = (
    #     interp_field(field_model.By, particle_state, axes, dy=eps)[0]
    #     - interp_field(field_model.By, particle_state, axes, dy=-eps)[0]
    # ) / (2 * eps)
    dBydz = (
        interp_field(field_model.By, particle_state, axes, dz=eps)[0]
        - interp_field(field_model.By, particle_state, axes, dz=-eps)[0]
    ) / (2 * eps)

    dBzdx = (
        interp_field(field_model.Bz, particle_state, axes, dx=eps)[0]
        - interp_field(field_model.Bz, particle_state, axes, dx=-eps)[0]
    ) / (2 * eps)
    dBzdy = (
        interp_field(field_model.Bz, particle_state, axes, dy=eps)[0]
        - interp_field(field_model.Bz, particle_state, axes, dy=-eps)[0]
    ) / (2 * eps)
    # dBzdz = (
    #     interp_field(field_model.Bz, particle_state, axes, dz=eps)[0]
    #     - interp_field(field_model.Bz, particle_state, axes, dz=-eps)[0]
    # ) / (2 * eps)

    # gyro-averaged equations of motion developed by Brizzard and Chan (Phys.
    # Plasmas 6, 4553, 1999),
    vpar = particle_state.vpar
    M = particle_state.magnetic_moment
    
    gamma = cp.sqrt(1 + 2 * B * M + vpar**2)
    pparl_B = vpar / B
    pparl_B2 = pparl_B / B
    Bxstar = Bx + pparl_B * (dBzdy - dBydz) - pparl_B2 * (Bz * dBdy - By * dBdz)
    Bystar = By + pparl_B * (dBxdz - dBzdx) - pparl_B2 * (Bx * dBdz - Bz * dBdx)
    Bzstar = Bz + pparl_B * (dBydx - dBxdy) - pparl_B2 * (By * dBdx - Bx * dBdy)
    Bsparl = (Bx * Bxstar + By * Bystar + Bz * Bzstar) / B
    gamma_Bsparl=1 / gamma / Bsparl
    pparl_gamma_Bsparl= vpar * gamma_Bsparl
    B_Bsparl = 1/ (B * Bsparl)
    M_gamma_Bsparl = M * gamma_Bsparl
    M_gamma_B_Bsparl = M_gamma_Bsparl / B

    # 	  ...now calculate dynamic quantities...
    dydt = cp.zeros((particle_state.x.size, 4))
    
    dydt[:, 0] = (
        pparl_gamma_Bsparl *Bxstar                     # curv drft + parl
        +M_gamma_B_Bsparl * (By * dBdz - Bz * dBdy)    # gradB drft
     	+B_Bsparl * (Ey * Bz - Ez * By)		       # ExB drft
    )
    dydt[:, 1] = (
        pparl_gamma_Bsparl * Bystar	               # curv drft + parl
     	+ M_gamma_B_Bsparl * (Bz * dBdx - Bx * dBdz)   # gradB drft
     	+ B_Bsparl * (Ez * Bx - Ex * Bz)               # ExB drft
    )
    dydt[:, 2] = (
        pparl_gamma_Bsparl * Bzstar		       # curv drft + parl
     	+M_gamma_B_Bsparl*(Bx*dBdy-By*dBdx)            # gradB drft
     	+B_Bsparl*(Ex*By-Ey*Bx)		               # ExB drft
    )
    dydt[:, 3] = (
        (Bxstar * Ex + Bystar * Ey + Bzstar * Ez) / Bsparl   # parl force
        - M_gamma_Bsparl * (Bxstar * dBdx+Bystar * dBdy + Bzstar * dBdz)
    )

    # calculate energy
    W = gamma - 1
    
    return dydt, B, W


    
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
