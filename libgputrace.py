import cupy as cp
import sys
from cupyx import jit
import math
from dataclasses import dataclass
from typing import Any
from astropy import constants, units
from line_profiler import profile

@dataclass
class TraceConfig:
    """Configuration for running the tracing code"""
    t_final: float            # end time of integration
    t_initial: float = 0      # start time of integration
    h_initial: float = .1     # initial step size
    h_min: float = .01        # min step size
    h_max: float = 1000       # max step size
    rtol: float = 1e-4        # relative tolerance
    grad_step: float = 5e-2   # finite diff delta step (half)
    output_freq: int = 10     # in iterations
    
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
    def initialize(cls, Bx, By, Bz, B, Ex, Ey, Ez, mass, charge):
        """Get a FieldModel() instance that is dimensionalized 
        and stored on the GPU.

        arrays are in shape (nt, nx, ny, nz)

        mass is not part of field model, but is used for 
        to redimensionalize.

        Input argument should have astropy units attached.
        Returns FieldModel instance         
        """
        q = charge
        Re = constants.R_earth
        c = constants.c
        sf = (q * Re / (mass * c**2))
        B_units = units.s / Re
        
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

    def interp(self, field, t, pos_x, pos_y, pos_z, axes, neighbors=None):
        """Interpolate a 3D gridded field at given positions.
        
        Args
          field: 3D gridded field to interpolate
          t: time to interpolate at
          pos_x, pos_y, pos_z: particle positions
          axes: rectilinear grid axes
          neighbors: optional, reuse this value for lookup of the neighbors
          dx, dy, dz: optional, perturb the position by these values    
        Return
          result: cupy array of interpolated field values at position
          neighbors: neighbors object for reuse
        """     
        if neighbors is None:
            neighbors = Neighbors(
                # side='right' helps first time step
                field_i=cp.searchsorted(axes.t, t, side='right'),
                field_j=cp.searchsorted(axes.x, pos_x),
                field_k=cp.searchsorted(axes.y, pos_y),
                field_l=cp.searchsorted(axes.z, pos_z),
            )

        result = cp.zeros(pos_x.shape)    
        arr_size = pos_x.size
        block_size = 128
        grid_size = int(math.ceil(arr_size / block_size))
        
        interp_quadlinear_kernel[grid_size, block_size](
            arr_size, result, field,
            neighbors.field_i,
            neighbors.field_j,
            neighbors.field_k,
            neighbors.field_l,
            t, pos_x, pos_y, pos_z,
            axes.t, axes.x, axes.y, axes.z
        )

        return result, neighbors
    

@dataclass
class ParticleState:
    """1D arrays of cartesian particle position component"""
    # these vary in time
    x: Any                    # x position
    y: Any                    # y position
    z: Any                    # z position
    ppar: Any                 # parallel momentum

    # these *don't* vary in time
    magnetic_moment: Any      # first invariant
    mass: Any                 # rest mass

    _dimensionalized: bool

    @classmethod
    def initialize(cls, x, y, z, ppar, magnetic_moment, mass, charge):
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
        
        gpu_x = cp.array((x / Re).to(1).value)
        gpu_y = cp.array((y / Re).to(1).value)
        gpu_z = cp.array((z / Re).to(1).value)
        gpu_ppar = cp.array((ppar / (c * mass)).to(1).value)
        gpu_magnetic_moment = cp.array(
            (magnetic_moment / (q * Re)).to(Re/units.s).value
        )
        gpu_mass = cp.array(mass.to(units.kg).value)
        
        return ParticleState(
            x=gpu_x, y=gpu_y, z=gpu_z, ppar=gpu_ppar,
            magnetic_moment=gpu_magnetic_moment, mass=gpu_mass,
            _dimensionalized=True        
        )
        
        
@dataclass
class Axes:
    """1D arrays of rectilinear grid axes"""
    t: Any                    # time axis
    x: Any                    # x axis
    y: Any                    # y axis
    z: Any                    # z axis
    _dimensionalized: bool
    
    @classmethod
    def initialize(cls, t, x, y, z):
        """Get Axes() instance that is dimensionalized and stored
        on the GPU.

        INput argument should have astropy units
        Returns Axes instance
        """        
        Re = constants.R_earth
        gpu_t = cp.array(redim_time(t.to(units.s).value))
        gpu_x = cp.array((x / Re).to(1).value)
        gpu_y = cp.array((y / Re).to(1).value)
        gpu_z = cp.array((z / Re).to(1).value)

        return Axes(
            t=gpu_t, x=gpu_x, y=gpu_y, z=gpu_z,
            _dimensionalized=True
        )
        
    
@dataclass
class ParticleHistory:
    """History of positions, parallel momentum, and useful values..
    """
    t: Any
    x: Any
    y: Any
    z: Any
    ppar: Any
    B: Any      # local field strength
    W: Any      # energy
    h: Any      # step size

        
@dataclass
class Neighbors:
    """Neighbors of given particles, used for interpolation"""
    field_i: Any
    field_j: Any
    field_k: Any
    field_l: Any
    

class RK45Coeffs:
    """
    Coefficients of the RK45 Algorithm
    """
    a2 = 0.25;
    a3 = 0.375;
    a4 = 12/13;
    a6 = 0.5;
    b21 = 0.25;
    b31 = 3/32;
    b32 = 9/32;
    b41 = 1932/2197;
    b42 = -7200/2197;
    b43 = 7296/2197;
    b51 = 439/216;
    b52 = -8;
    b53 = 3680/513;
    b54 = -845/4104;
    b61 = -8/27;
    b62 = 2;
    b63 = -3544/2565;
    b64 = 1859/4104;
    b65 = -11/40;
    c1 = 25/216;
    c3 = 1408/2565;
    c4 = 2197/4104;
    c5 = -0.20;
    d1 = 1/360;
    d3 = -128/4275;
    d4 = -2197/75240;
    d5 = 0.02;
    d6 = 2/55;
    

@profile
def trace_trajectory(config, particle_state, field_model, axes):
    """Perform a euler integration particle trace.
    
    Works on a rectilinear grid.

    Args
      config: instance of libgputrace.TraceConfig
      particle_state: instance of libgputrace.ParticleState
      field_model: instance of libgputrace.FieldModel
      axes: instance of libgputrace.Axes
    Returns  
      hist: instance of libgputrace.ParticleHistory
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

    # This implements the RK45 adaptive integration algorithm, with
    # absolute/relative tolerance and minimum/maximum step sizes
    t = cp.ones(particle_state.x.size) * redim_time(config.t_initial)
    y = cp.zeros((particle_state.x.size, 5))
    y[:, 0] = particle_state.x
    y[:, 1] = particle_state.y
    y[:, 2] = particle_state.z
    y[:, 3] = particle_state.ppar
    y[:, 4] = particle_state.magnetic_moment

    h = cp.ones(particle_state.x.size) * config.h_initial
    h_min = cp.ones(particle_state.x.size) * config.h_min
    h_max = cp.ones(particle_state.x.size) * config.h_max

    t_final = redim_time(config.t_final)
    all_complete = False
    R = RK45Coeffs
    iter_count = 0

    hist_y = []
    hist_t = []
    hist_W = []
    hist_B = []
    hist_h = []

    while not all_complete:
        h = cp.minimum(h_max, cp.maximum(h, h_min))
        h_ = cp.zeros((h.size, 5))             # cupy broadcasting workaround

        for i in range(5):
            h_[:, i] = h

        k1 = h_ * rhs(
            t,
            y,
            field_model, axes, config
        )
        k2 = h_ * rhs(
            t + h * R.a2,
            y + h_ * R.b21 * k1, field_model, axes, config
        )
        k3 = h_ * rhs(
            t + h * R.a3,
            y + h_ * (R.b31*k1 + R.b32*k2), field_model, axes, config
        )
        k4 = h_ * rhs(
            t + h * R.a4,
            y + h_ * (R.b41 * k1 + R.b42 * k2 + R.b43 * k3),
            field_model, axes, config
        )
        k5 = h_ * rhs(
            t + h,
            y + h_ * (R.b51 * k1 + R.b52 * k2 + R.b53 * k3 + R.b54 * k4),
            field_model, axes, config
        )
        k6 = h_ * rhs(
            t + h * R.a6,
            y + h_ * (R.b61 * k1 + R.b62 * k2 + R.b63 * k3 + R.b64 * k4 + R.b65 * k5),
            field_model, axes, config
        )
                
        y_next = y + R.c1*k1 + R.c3*k3 + R.c4*k4 + R.c5*k5
        z_next = y + R.d1*k1 + R.d3*k3 + R.d4*k4 + R.d5*k5 + R.d6*k6
        err = cp.linalg.norm(y_next - z_next, axis=1)        

        ymag = cp.linalg.norm(y, axis=1)
        tolerance = config.rtol * ymag
        scale = 0.84*(tolerance/err)**(1/4)
        mask = (err < config.rtol * ymag) & (t < t_final)

        dt = h.copy()
        dt[~mask] = 0        
        t += dt
        y[mask] = y_next[mask]
        h = h * scale
        all_complete = cp.all(t >= t_final)
        iter_count += 1

        # Save incremented particles to history
        if iter_count % config.output_freq == 0:
            tmp_B, _ = field_model.interp(
                field_model.B, t, y[:, 0], y[:, 1], y[:, 2], axes
            )
            tmp_gamma = cp.sqrt(
                1 + 2 * tmp_B * y[:, 4] + y[:, 3]**2
            )
            tmp_W = tmp_gamma - 1
            
            hist_t.append(t.get())
            hist_y.append(y.get())
            hist_B.append(tmp_B.get())
            hist_W.append(tmp_W.get())
            hist_h.append(h.get())

        print(f'Complete: {100 * min(t.min() / t_final, 1):.1f}% '
              f'(iter {iter_count}, {mask.sum()} iterated, h mean '
              f'{h.mean():.1E})')
        
        sys.stdout.flush()

    print(f'Took {iter_count} iterations')

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
        t=hist_t, x=hist_pos_x, y=hist_pos_y, z=hist_pos_z,
        ppar=hist_ppar, B=hist_B, W=hist_W, h=hist_h,
    )


@profile
def rhs(t, y, field_model, axes, config):
    """RIght hand side of the guiding center equation differential equation.

    Args
      t: ODE derivative variable
      y: ODE state variable
      field_model: instance of FieldModel (provides E and B fields)
      axes: instance of Axes (rectilinear grid axes)
      config: instance of Config (tracing configuration)
    Returns
      dydt: cupy array (nparticles, 4). First three columns are position, fourth
        is parallel momentum
    """
    pos_x = y[:, 0]
    pos_y = y[:, 1]
    pos_z = y[:, 2]            
 
    # Get B and E at particle position
    Bx, neighbors = field_model.interp(
        field_model.Bx, t, pos_x, pos_y, pos_z, axes
    )
    By, _ = field_model.interp(
        field_model.By, t, pos_x, pos_y, pos_z, axes, neighbors=neighbors
    )
    Bz, _ = field_model.interp(
        field_model.Bz, t, pos_x, pos_y, pos_z, axes, neighbors=neighbors
    )
    B, _ = field_model.interp(
        field_model.B, t, pos_x, pos_y, pos_z, axes, neighbors=neighbors
    )
    Ex, _ = field_model.interp(
        field_model.Ex, t, pos_x, pos_y, pos_z, axes, neighbors=neighbors
    )
    Ey, _ = field_model.interp(
        field_model.Ey, t, pos_x, pos_y, pos_z, axes, neighbors=neighbors
    )
    Ez, _ = field_model.interp(
        field_model.Ez, t, pos_x, pos_y, pos_z, axes, neighbors=neighbors
    )
    
    # Get derivatives from finite difference
    # ---------------------------------------
    eps = config.grad_step
    
    # in |B| magnitude
    pos_x_forw = pos_x + eps
    pos_x_back = pos_x - eps
    dBdx_forw, dx_forw_neighbors = field_model.interp(
        field_model.B, t, pos_x_forw, pos_y, pos_z, axes
    )
    dBdx_back, dx_back_neighbors = field_model.interp(
        field_model.B, t, pos_x_back, pos_y, pos_z, axes
    )
    dBdx = (dBdx_forw - dBdx_back) / (2 * eps)

    pos_y_forw = pos_y + eps
    pos_y_back = pos_y - eps    
    dBdy_forw, dy_forw_neighbors = field_model.interp(
        field_model.B, t, pos_x, pos_y_forw, pos_z, axes
    )
    dBdy_back, dy_back_neighbors = field_model.interp(
        field_model.B, t, pos_x, pos_y_back, pos_z, axes
    )
    dBdy = (dBdy_forw - dBdy_back) / (2 * eps)

    pos_z_forw = pos_z + eps
    pos_z_back = pos_z - eps    

    dBdz_forw, dz_forw_neighbors = field_model.interp(
        field_model.B, t, pos_x, pos_y, pos_z_forw, axes
    )
    dBdz_back, dz_back_neighbors = field_model.interp(
        field_model.B, t, pos_x, pos_y, pos_z_back, axes
    )
    dBdz = (dBdz_forw - dBdz_back) / (2 * eps)

    # in Bx
    dBxdy_forw, _ = field_model.interp(
        field_model.Bx, t, pos_x, pos_y_forw, pos_z, axes,
        neighbors=dy_forw_neighbors
    )
    dBxdy_back, _ = field_model.interp(
        field_model.Bx, t, pos_x, pos_y_back, pos_z, axes,
        neighbors=dy_back_neighbors
    )
    dBxdy = (dBxdy_forw - dBxdy_back) / (2 * eps)

    dBxdz_forw, _ = field_model.interp(
        field_model.Bx, t, pos_x, pos_y, pos_z_forw, axes,
        neighbors=dz_forw_neighbors
    )
    dBxdz_back, _ = field_model.interp(
        field_model.Bx, t, pos_x, pos_y, pos_z_back, axes,
        neighbors=dz_back_neighbors
    )
    dBxdz = (dBxdz_forw - dBxdz_back) / (2 * eps)
    
    # in By
    dBydx_forw, _ = field_model.interp(
        field_model.By, t, pos_x_forw, pos_y, pos_z, axes,
        neighbors=dx_forw_neighbors
    )
    dBydx_back, _ = field_model.interp(
        field_model.By, t, pos_x_back, pos_y, pos_z, axes,
        neighbors=dx_back_neighbors
    )
    dBydx = (dBydx_forw - dBydx_back) / (2 * eps)
    
    dBydz_forw, _ = field_model.interp(
        field_model.By, t, pos_x, pos_y, pos_z_forw, axes,
        neighbors=dz_forw_neighbors
    )
    dBydz_back, _ = field_model.interp(
        field_model.By, t, pos_x, pos_y, pos_z_back, axes, 
        neighbors=dz_back_neighbors
    )
    dBydz = (dBydz_forw - dBydz_back) / (2 * eps)

    # in Bz
    dBzdx_forw, _ = field_model.interp(
        field_model.Bz, t, pos_x_forw, pos_y, pos_z, axes,
        neighbors=dx_forw_neighbors
    )
    dBzdx_back, _ = field_model.interp(
        field_model.Bz, t, pos_x_back, pos_y, pos_z, axes,
        neighbors=dx_back_neighbors
    )
    dBzdx = (dBzdx_forw - dBzdx_back) / (2 * eps)
    
    dBzdy_forw, _ = field_model.interp(
        field_model.Bz, t, pos_x, pos_y_forw, pos_z, axes,
        neighbors=dy_forw_neighbors
    )
    dBzdy_back, _ = field_model.interp(
        field_model.Bz, t, pos_x, pos_y_back, pos_z, axes,
        neighbors=dy_back_neighbors
    )
    dBzdy = (dBzdy_forw - dBzdy_back) / (2 * eps)

    # Launch Kernel to handle rest of RHS
    # --------------------------------------
    arr_size = pos_x.size
    block_size = 256
    grid_size = int(math.ceil(arr_size / block_size))

    dydt = cp.zeros((pos_x.size, 5))
    
    rhs_kernel[grid_size, block_size](
        arr_size, y, dydt, 
        Bx, By, Bz, B,
        Ex, Ey, Ez,
        dBdx, dBdy, dBdz,
        dBxdy, dBxdz, dBydx, dBydz, dBzdx, dBzdy,
    )

    return dydt


@jit.rawkernel()
def rhs_kernel(
        arr_size, y, dydt_arr, 
        Bx_arr, By_arr, Bz_arr, B_arr,
        Ex_arr, Ey_arr, Ez_arr,
        dBdx_arr, dBdy_arr, dBdz_arr,
        dBxdy_arr, dBxdz_arr, dBydx_arr, dBydz_arr, dBzdx_arr, dBzdy_arr,
):
    """[CUPY KERNEL] implements RHS of oDE. 

    Uses gyro-averaged equations of motion developed by Brizzard and Chan (Phys.
    Plasmas 6, 4553, 1999),

    Writes output to dydt[idx, :]
    """
    idx = jit.blockDim.x * jit.blockIdx.x + jit.threadIdx.x

    if idx < arr_size:    
        # Pull variables out of arrays
        ppar  = y[idx, 3]
        M     = y[idx, 4]

        Bx = Bx_arr[idx]
        By = By_arr[idx]
        Bz = Bz_arr[idx]
        B  = B_arr[idx]

        Ex = Ex_arr[idx]
        Ey = Ey_arr[idx]
        Ez = Ez_arr[idx]

        dBdx = dBdx_arr[idx]
        dBdy = dBdy_arr[idx]
        dBdz = dBdz_arr[idx]
    
        dBxdy = dBxdy_arr[idx]
        dBxdz = dBxdz_arr[idx]
        dBydx = dBydx_arr[idx]
        dBydz = dBydz_arr[idx]
        dBzdx = dBzdx_arr[idx]
        dBzdy = dBzdy_arr[idx]
        
        # gyro-averaged equations of motion developed by Brizzard and Chan (Phys.
        # Plasmas 6, 4553, 1999),
        gamma = cp.sqrt(1 + 2 * B * M + ppar**2)
        pparl_B = ppar / B
        pparl_B2 = pparl_B / B
        Bxstar = Bx + pparl_B * (dBzdy - dBydz) - pparl_B2 * (Bz * dBdy - By * dBdz)
        Bystar = By + pparl_B * (dBxdz - dBzdx) - pparl_B2 * (Bx * dBdz - Bz * dBdx)
        Bzstar = Bz + pparl_B * (dBydx - dBxdy) - pparl_B2 * (By * dBdx - Bx * dBdy)
        Bsparl = (Bx * Bxstar + By * Bystar + Bz * Bzstar) / B
        gamma_Bsparl = 1 / (gamma * Bsparl)
        pparl_gamma_Bsparl = ppar * gamma_Bsparl
        B_Bsparl = 1/ (B * Bsparl)
        M_gamma_Bsparl = M * gamma_Bsparl
        M_gamma_B_Bsparl = M_gamma_Bsparl / B
        
        # 	  ...now calculate dynamic quantities...
        dydt_arr[idx, 0] = (
            pparl_gamma_Bsparl * Bxstar                    # curv drft + parl
            + M_gamma_B_Bsparl * (By * dBdz - Bz * dBdy)   # gradB drft
     	    + B_Bsparl * (Ey * Bz - Ez * By)	           # ExB drft
        )
        dydt_arr[idx, 1] = (
            pparl_gamma_Bsparl * Bystar	                   # curv drft + parl
     	    + M_gamma_B_Bsparl * (Bz * dBdx - Bx * dBdz)   # gradB drft
     	    + B_Bsparl * (Ez * Bx - Ex * Bz)               # ExB drft
        )
        dydt_arr[idx, 2] = (
            pparl_gamma_Bsparl * Bzstar		           # curv drft + parl
     	    + M_gamma_B_Bsparl * (Bx * dBdy - By * dBdx)   # gradB drft
     	    + B_Bsparl * (Ex * By - Ey * Bx)		   # ExB drft
        )
        dydt_arr[idx, 3] = (
            (Bxstar * Ex + Bystar * Ey + Bzstar * Ez) / Bsparl   # parl force
            - M_gamma_Bsparl * (Bxstar * dBdx + Bystar * dBdy + Bzstar * dBdz)
        )
        

@jit.rawkernel()
def interp_quadlinear_kernel(
        arr_size, result, field,
        field_i, field_j, field_k, field_l,
        t, pos_x, pos_y, pos_z,
        t_axis, x_axis, y_axis, z_axis
):
    """[CUPY KERNEL] Interpolate field using neighbors.
    
    Uses qudlinear interpolation.
    """
    # Note to self: tried weighted sum of nearest neighbors
    # where weight was inverse difference, but result was
    # less smooth
    idx = jit.blockDim.x * jit.blockIdx.x + jit.threadIdx.x
    
    if idx < arr_size:
        # Find indices of the surrounding grid points
        t0, t1 = field_i[idx] - 1, field_i[idx]
        x0, x1 = field_j[idx] - 1, field_j[idx]
        y0, y1 = field_k[idx] - 1, field_k[idx]
        z0, z1 = field_l[idx] - 1, field_l[idx]
        
        # Get the surrounding values
        c0000 = field[t0, x0, y0, z0]
        c0001 = field[t0, x0, y0, z1]
        c0010 = field[t0, x0, y1, z0]
        c0011 = field[t0, x0, y1, z1]
        c0100 = field[t0, x1, y0, z0]
        c0101 = field[t0, x1, y0, z1]
        c0110 = field[t0, x1, y1, z0]
        c0111 = field[t0, x1, y1, z1]
        c1000 = field[t1, x0, y0, z0]
        c1001 = field[t1, x0, y0, z1]
        c1010 = field[t1, x0, y1, z0]
        c1011 = field[t1, x0, y1, z1]
        c1100 = field[t1, x1, y0, z0]
        c1101 = field[t1, x1, y0, z1]
        c1110 = field[t1, x1, y1, z0]
        c1111 = field[t1, x1, y1, z1]
    
        # Compute interpolation weights
        epsilon = 1e-10
        td = (t[idx] - t_axis[t0]) / (t_axis[t1] - t_axis[t0] + epsilon)
        xd = (pos_x[idx] - x_axis[x0]) / (x_axis[x1] - x_axis[x0] + epsilon)
        yd = (pos_y[idx] - y_axis[y0]) / (y_axis[y1] - y_axis[y0] + epsilon)
        zd = (pos_z[idx] - z_axis[z0]) / (z_axis[z1] - z_axis[z0] + epsilon)
        
        # Interpolation in 4D
        c00 = c0000 * (1 - xd) + c0100 * xd
        c01 = c0001 * (1 - xd) + c0101 * xd
        c10 = c0010 * (1 - xd) + c0110 * xd
        c11 = c0011 * (1 - xd) + c0111 * xd
        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd
        c00 = c1000 * (1 - xd) + c1100 * xd
        c01 = c1001 * (1 - xd) + c1101 * xd
        c10 = c1010 * (1 - xd) + c1110 * xd
        c11 = c1011 * (1 - xd) + c1111 * xd
        c2 = c00 * (1 - yd) + c10 * yd
        c3 = c01 * (1 - yd) + c11 * yd
        c4 = c0 * (1 - zd) + c2 * zd
        c5 = c1 * (1 - zd) + c3 * zd

        result[idx] = c4 * (1 - td) + c5 * td


def redim_time(val):
    """Redimensionalize a time value.

    Args
      value: value in seconds
    Returns
      value in redimensionalized units
    """
    sf = constants.c / constants.R_earth
    return (sf * val * units.s).to(1).value

