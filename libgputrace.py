from dataclasses import dataclass
import math
import sys
from typing import Any, Optional, List, Callable

from astropy import constants, units
from astropy.units import Quantity
import cupy as cp
from cupyx import jit
import numpy as np


@dataclass
class TraceConfig:
    """Configuration for running the tracing code"""
    t_final: Quantity                                    # end time of integration
    output_freq: Optional[int] = None                    # in iterations
    stopping_conditions: Optional[List[Callable]] = None # stopping conditions
    t_initial: Quantity = 0 * units.s                    # start time of integration
    h_initial: Quantity = 1 * units.ms                   # initial step size
    h_min: Quantity = .1 * units.ms                      # min step size
    h_max: Quantity = 1 * units.s                        # max step size
    rtol: float = 1e-2                                   # relative tolerance

    
class FieldModel:
    """Abstract Base class of magnetic and electric field models.

    Attributes
      Bx: Magnetic Field X (dimensionalized)
      By: Magnetic Field Y (dimensionalized)
      Bz: Magnetic Field Z (dimensionalized)
      B: Magnetic Field Magnitude (dimensionalized)
      Ex: Electric Field X (dimensionalized)
      Ey: Electric Field Y (dimensionalized)
      Ez: Electric Field Z (dimensionalized)
    """    
    DEFAULT_RAW_B0 = 31e3 * units.nT

    def __init__(self, Bx, By, Bz, Ex, Ey, Ez, mass, charge, axes, B0=DEFAULT_RAW_B0):
        """Get an instance that is dimensionalized and stored on the GPU.

        mass is not part of field model, but is used to redimensionalize.

        Input argument should have astropy units attached.
        Returns instance of class 
        """
        q = charge
        Re = constants.R_earth
        c = constants.c
        sf = (q * Re / (mass * c**2))
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

    def multi_interp(self, t, y):
        """Interpolate field values at given positions.
        
        Args
          t: vector of particle times
          y: vector of hsape (npart, 5) of particle states
        Return
           Bx, By, Bz, Ex, Ey, Ez,
           dBxdx, dBxdy, dBxdz, 
           dBydx, dBydy, dBydz, 
           dBzdx, dBzdy, dBzdz, 
           B, dBdx, dBdy, dBdz
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

        bxvec = self.Bx.reshape(nttl, order='F')
        byvec = self.By.reshape(nttl, order='F')
        bzvec = self.Bz.reshape(nttl, order='F')
        exvec = self.Ex.reshape(nttl, order='F')
        eyvec = self.Ey.reshape(nttl, order='F')
        ezvec = self.Ez.reshape(nttl, order='F')

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
        grid_size = int(math.ceil(arr_size / block_size))

        multi_interp_kernel[grid_size, block_size](
            arr_size, nx, ny, nz, nt, nxy, nxyz, nttl,
            ix, iy, iz, it,
            t, y, b0, r_inner,
            t_axis, x_axis, y_axis, z_axis, 
            bxvec, byvec, bzvec, exvec, eyvec, ezvec,
            bx, by, bz, ex, ey, ez,
            dbxdx, dbxdy, dbxdz, 
            dbydx, dbydy, dbydz, 
            dbzdx, dbzdy, dbzdz, 
            b, dbdx, dbdy, dbdz,
        )
        
        # Return values as tuple
        return (
            bx, by, bz, ex, ey, ez,
            dbxdx, dbxdy, dbxdz, 
            dbydx, dbydy, dbydz, 
            dbzdx, dbzdy, dbzdz, 
            b, dbdx, dbdy, dbdz,
        )
    

class ParticleState:
    """1D arrays of cartesian particle position component"

    Attributes
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
        self.magnetic_moment = cp.array(
            (magnetic_moment / (q * Re)).to(Re/units.s).value
        )
        self.mass = cp.array(mass.to(units.kg).value)
        
        
class Axes:
    """1D arrays of rectilinear grid axes

    Attributes
      t: time axis
      x: x axis
      y: y axis
      z: z axis
    """
    def __init__(self, t, x, y, z, r_inner):
        """Initialize instance that is dimensionalized and stored
        on the GPU.

        Input arguments should have astropy units
        """
        assert len(t.shape) == 1
        assert len(x.shape) == 1
        assert len(y.shape) == 1
        assert len(z.shape) == 1
        
        Re = constants.R_earth
        self.t = cp.array(redim_time(t))
        self.x = cp.array((x / Re).to(1).value)
        self.y = cp.array((y / Re).to(1).value)
        self.z = cp.array((z / Re).to(1).value)
        self.r_inner = (r_inner / Re).to(1).value

    def get_neighbors(self, t, pos_x, pos_y, pos_z):
        """Get instance of RectilinearNeighbors specifying surrounding
        cell through indeces of upper corner

        Returns instance of RectilinearNeighbors
        """
        return RectilinearNeighbors(
            # side='right' helps first time step
            field_i=cp.searchsorted(self.x, pos_x) ,
            field_j=cp.searchsorted(self.y, pos_y) ,
            field_k=cp.searchsorted(self.z, pos_z) ,
            field_l=cp.searchsorted(self.t, t, side='right'),
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
class RectilinearNeighbors:
    """Neighbors of given particles, used for interpolation"""
    field_i: Any
    field_j: Any
    field_k: Any
    field_l: Any
    

class RK45Coeffs:
    """
    Coefficients of the RK45 Algorithm
    """
    # Cash-Karp Coefficients
    a2 = 1/5
    a3 = 3/10
    a4 = 3/5
    a5 = 1
    a6 = 7/8

    b21 = 1/5
    b31 = 3/40 
    b32 = 9/40
    b41 = 3/10
    b42 = -9/10
    b43 = 6/5
    b51 = -11/54
    b52 = 5/2
    b53 = -70/27
    b54 = 35/27
    b61 = 1631/55296
    b62 = 175/512
    b63 = 575/13824
    b64 = 44275/110592
    b65 = 253/4096

    c1 = 37/378
    c3 = 250/621
    c4 = 125/594
    c5 = 0
    c6 = 512/1771
    
    d1 = 2825/27648
    d3 = 18575/48384
    d4 = 13525/55296
    d5 = 277/14336
    d6 = 1/4
    

def trace_trajectory(config, particle_state, field_model):
    """Perform a euler integration particle trace.
    
    Works on a rectilinear grid.

    Args
      config: instance of libgputrace.TraceConfig
      particle_state: instance of libgputrace.ParticleState
      field_model: instance of libgputrace.FieldModel
    Returns  
      hist: instance of libgputrace.ParticleHistory
    """
    # This implements the RK45 adaptive integration algorithm, with
    # absolute/relative tolerance and minimum/maximum step sizes
    npart = particle_state.x.size
    t = cp.zeros(npart) + redim_time(config.t_initial)
    
    y = cp.zeros((particle_state.x.size, 5))
    y[:, 0] = particle_state.x
    y[:, 1] = particle_state.y
    y[:, 2] = particle_state.z
    y[:, 3] = particle_state.ppar
    y[:, 4] = particle_state.magnetic_moment

    h = cp.zeros(npart) + redim_time(config.h_initial)
    h_min = cp.zeros(npart) + redim_time(config.h_min)
    h_max = cp.zeros(npart) + redim_time(config.h_max)

    t_final = redim_time(config.t_final)
    all_complete = False
    stopped = cp.zeros(npart, dtype=bool)

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

        # Call rhs() function to implement multiple function evaluations of
        # right hand side.
        k1, B = rhs(
            t,
            y,
            field_model, config
        )        
        k2, _ = rhs(
            t + h * R.a2,
            y + h_ * R.b21 * k1,
            field_model, config
        )
        k3, _ = rhs(
            t + h * R.a3,
            y + h_ * (R.b31 * k1 + R.b32 * k2),
            field_model, config
        )
        k4, _ = rhs(
            t + h * R.a4,
            y + h_ * (R.b41 * k1 + R.b42 * k2 + R.b43 * k3),
            field_model, config
        )
        k5, _ = rhs(
            t + h * R.a5,
            y + h_ * (R.b51 * k1 + R.b52 * k2 + R.b53 * k3 + R.b54 * k4),
            field_model, config
        )
        k6, _ = rhs(
            t + h * R.a6,
            y + h_ * (R.b61 * k1 + R.b62 * k2 + R.b63 * k3 + R.b64 * k4 + R.b65 * k5),
            field_model, config
        )
            
        k1 *= h_
        k2 *= h_
        k3 *= h_
        k4 *= h_
        k5 *= h_
        k6 *= h_

        # Save incremented particles to history
        if (config.output_freq is not None
            and (iter_count % config.output_freq == 0)):
            gamma = cp.sqrt(1 + 2 * B * y[:, 4] + y[:, 3]**2)
            W = gamma - 1            
            hist_t.append(t.get())
            hist_y.append(y.get())
            hist_B.append(B.get())
            hist_W.append(W.get())
            hist_h.append(h.get())

        num_iterated = do_step(
            k1, k2, k3, k4, k5, k6, y, h, t, config.rtol, t_final,
            field_model, stopped, config.stopping_conditions,
        )
        all_complete = cp.all(stopped)
        iter_count += 1

        r_mean = cp.sqrt(y[:, 0]**2 + y[:, 1]**2 + y[:, 2]**2).mean()
        h_step = undim_time(float(h.mean())).to(units.ms).value
        
        print(f'Complete: {100 * min(t.min() / t_final, 1):.1f}% '
              f'(iter {iter_count}, {num_iterated} iterated, h mean '
              f'{h_step:.2f} ms, r mean {r_mean:.2f})')
        
        sys.stdout.flush()

    print(f'Took {iter_count} iterations')

    # Always save last step of each, even if not recording full history
    gamma = cp.sqrt(1 + 2 * B * y[:, 4] + y[:, 3]**2)
    W = gamma - 1            
    hist_t.append(t.get())
    hist_y.append(y.get())
    hist_B.append(B.get())
    hist_W.append(W.get())
    hist_h.append(h.get())    
    
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


def rhs(t, y, field_model, config):
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
    # Get B Values
    (
        Bx, By, Bz, Ex, Ey, Ez,
        dBxdx, dBxdy, dBxdz, 
        dBydx, dBydy, dBydz, 
        dBzdx, dBzdy, dBzdz, 
        B, dBdx, dBdy, dBdz
    ) = (
        field_model.multi_interp(t, y)
    )

    # need to account for dimensionalization of magnitude
    if field_model.negative_charge:
        B *= -1
        dBdx *= -1
        dBdy *= -1
        dBdz *= -1
    
    # Launch Kernel to handle rest of RHS
    arr_size = y.shape[0]
    block_size = 256
    grid_size = int(math.ceil(arr_size / block_size))

    r_inner = cp.zeros(arr_size) + field_model.axes.r_inner
    dydt = cp.zeros((arr_size, 5))

    rhs_kernel[grid_size, block_size](
        arr_size, y, t, 
        Bx, By, Bz, B,
        Ex, Ey, Ez,
        field_model.axes.x,
        field_model.axes.y,
        field_model.axes.z,
        field_model.axes.t,
        field_model.axes.x.size,
        field_model.axes.y.size,
        field_model.axes.z.size,
        field_model.axes.t.size,
        r_inner,
        dBdx, dBdy, dBdz,
        dBxdy, dBxdz, dBydx, dBydz, dBzdx, dBzdy,
        dydt,
    )

    return dydt, B


@jit.rawkernel()
def rhs_kernel(
        arr_size, y, t, 
        Bx_arr, By_arr, Bz_arr, B_arr,
        Ex_arr, Ey_arr, Ez_arr,
        x_axis, y_axis, z_axis, t_axis, nx, ny, nz, nt, r_inner,
        dBdx_arr, dBdy_arr, dBdz_arr,
        dBxdy_arr, dBxdz_arr, dBydx_arr, dBydz_arr, dBzdx_arr, dBzdy_arr,
        dydt_arr, 
):
    """[CUPY KERNEL] implements RHS of oDE. 

    Code adapted from Fortran. Uses gyro-averaged equations of motion 
    developed by Brizzard and Chan (Phys. Plasmas 6, 4553, 1999),

    Writes output to dydt[idx, :]
    """
    idx = jit.blockDim.x * jit.blockIdx.x + jit.threadIdx.x

    if idx < arr_size and not (
            # Out of bounds check
            y[idx, 0] < x_axis[0] or y[idx, 0] > x_axis[nx - 1] or
            y[idx, 1] < y_axis[0] or y[idx, 1] > y_axis[ny - 1] or
            y[idx, 2] < z_axis[0] or y[idx, 2] > z_axis[nz - 1] or
            t[idx] < t_axis[0] or t[idx] > t_axis[nt - 1] or
            (y[idx, 0]**2 + y[idx, 1]**2 + y[idx, 2]**2)**0.5 < r_inner[idx]
    ):    
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
def multi_interp_kernel(
        arr_size, nx, ny, nz, nt, nxy, nxyz, nttl,
        ix, iy, iz, it,
        t, y, b0, r_inner,
        t_axis, x_axis, y_axis, z_axis, 
        bxvec, byvec, bzvec, exvec, eyvec, ezvec,
        bx, by, bz, ex, ey, ez,
        dbxdx, dbxdy, dbxdz, 
        dbydx, dbydy, dbydz, 
        dbzdx, dbzdy, dbzdz, 
        b, dbdx, dbdy, dbdz,
):
    """[CUPY KERNEL] Four dimensional interpolation with finite differencing
    and reused weights.

    Code adapted from fortran.
    """
    idx = jit.blockDim.x * jit.blockIdx.x + jit.threadIdx.x
    
    if idx < arr_size and not (
            # Out of bounds check
            y[idx, 0] < x_axis[0] or y[idx, 0] > x_axis[nx - 1] or
            y[idx, 1] < y_axis[0] or y[idx, 1] > y_axis[ny - 1] or
            y[idx, 2] < z_axis[0] or y[idx, 2] > z_axis[nz - 1] or
            t[idx] < t_axis[0] or t[idx] > t_axis[nt - 1] or
            (y[idx, 0]**2 + y[idx, 1]**2 + y[idx, 2]**2)**0.5 < r_inner[idx]
    ):            
        dx = x_axis[ix[idx] + 1] - x_axis[ix[idx]]
        dy = y_axis[iy[idx] + 1] - y_axis[iy[idx]]
        dz = z_axis[iz[idx] + 1] - z_axis[iz[idx]]
        dt = t_axis[it[idx] + 1] - t_axis[it[idx]]

	# ...determine memory location corresponding to ix,iy,iz,it
	#	and ix+1,iy+1,iz+1,it+1...        
        jjjj= ix[idx] + iy[idx]*nx + iz[idx]*nxy + it[idx]*nxyz + 1          
        ijjj= jjjj - 1
        jijj= jjjj - nx
        iijj= jijj - 1
        jjij= jjjj - nxy
        ijij= jjij - 1
        jiij= jijj - nxy
        iiij= jiij - 1
        jjji= jjjj - nxyz
        ijji= ijjj - nxyz
        jiji= jijj - nxyz
        iiji= iijj - nxyz
        jjii= jjij - nxyz
        ijii= ijij - nxyz
        jiii= jiij - nxyz
        iiii= iiij - nxyz

        #...calculate weighting factors...
        w1=cp.abs(y[idx, 0] - x_axis[ix[idx]])/dx
        w1m=1.-w1
        w2=cp.abs(y[idx, 1] - y_axis[iy[idx]])/dy
        w2m=1.-w2
        w3=cp.abs(y[idx, 2] - z_axis[iz[idx]])/dz
        w3m=1.-w3
        w4=cp.abs(t[idx] - t_axis[it[idx]])/dt
        w4m=1.-w4
        
        w1m2m=w1m*w2m
        w12m=w1*w2m
        w12=w1*w2
        w1m2=w1m*w2
        
        w1m2m3m=w1m2m*w3m
        w12m3m=w12m*w3m
        w123m=w12*w3m
        w1m23m=w1m2*w3m
        
        w1m2m3=w1m2m*w3
        w12m3=w12m*w3
        w123=w12*w3
        w1m23=w1m2*w3
        
        ww01=w1m2m3m*w4m
        ww02=w12m3m*w4m
        ww03=w123m*w4m
        ww04=w1m23m*w4m
        ww05=w1m2m3*w4m
        ww06=w12m3*w4m
        ww07=w123*w4m
        ww08=w1m23*w4m
        ww09=w1m2m3m*w4
        ww10=w12m3m*w4
        ww11=w123m*w4
        ww12=w1m23m*w4
        ww13=w1m2m3*w4
        ww14=w12m3*w4
        ww15=w123*w4
        ww16=w1m23*w4
        
        #..define some factors often repeated in the interpolations...
        r = (y[idx, 0]**2 + y[idx, 1]**2 + y[idx, 2]**2)**(0.5)
        r2=r*r
        bfac1=3.*b0[idx]/r2/r2/r
        bfac2=5.*bfac1/r2
        
        # ...interpolate field components...
        bx[idx]=(
            bxvec[iiii]*ww01+bxvec[jiii]*ww02+bxvec[jjii]*ww03
            +bxvec[ijii]*ww04+bxvec[iiji]*ww05+bxvec[jiji]*ww06
            +bxvec[jjji]*ww07+bxvec[ijji]*ww08+bxvec[iiij]*ww09
            +bxvec[jiij]*ww10+bxvec[jjij]*ww11+bxvec[ijij]*ww12
            +bxvec[iijj]*ww13+bxvec[jijj]*ww14+bxvec[jjjj]*ww15
            +bxvec[ijjj]*ww16 
            -bfac1*y[idx, 0]*y[idx, 2]
        )

        by[idx]=(
            byvec[iiii]*ww01+byvec[jiii]*ww02+byvec[jjii]*ww03
            +byvec[ijii]*ww04+byvec[iiji]*ww05+byvec[jiji]*ww06
            +byvec[jjji]*ww07+byvec[ijji]*ww08+byvec[iiij]*ww09
            +byvec[jiij]*ww10+byvec[jjij]*ww11+byvec[ijij]*ww12
            +byvec[iijj]*ww13+byvec[jijj]*ww14+byvec[jjjj]*ww15
            +byvec[ijjj]*ww16 
            -bfac1*y[idx, 1]*y[idx, 2]
        )
        
        bz[idx]=(
            bzvec[iiii]*ww01+bzvec[jiii]*ww02+bzvec[jjii]*ww03
            +bzvec[ijii]*ww04+bzvec[iiji]*ww05+bzvec[jiji]*ww06
            +bzvec[jjji]*ww07+bzvec[ijji]*ww08+bzvec[iiij]*ww09
            +bzvec[jiij]*ww10+bzvec[jjij]*ww11+bzvec[ijij]*ww12
            +bzvec[iijj]*ww13+bzvec[jijj]*ww14+bzvec[jjjj]*ww15
            +bzvec[ijjj]*ww16 
            -bfac1*y[idx, 2]*y[idx, 2] + b0[idx]/r2/r
        )

        ex[idx]=(
            exvec[iiii]*ww01+exvec[jiii]*ww02+exvec[jjii]*ww03
            +exvec[ijii]*ww04+exvec[iiji]*ww05+exvec[jiji]*ww06
            +exvec[jjji]*ww07+exvec[ijji]*ww08+exvec[iiij]*ww09
            +exvec[jiij]*ww10+exvec[jjij]*ww11+exvec[ijij]*ww12
            +exvec[iijj]*ww13+exvec[jijj]*ww14+exvec[jjjj]*ww15
            +exvec[ijjj]*ww16
        )
        
        ey[idx]=(
            eyvec[iiii]*ww01+eyvec[jiii]*ww02+eyvec[jjii]*ww03
            +eyvec[ijii]*ww04+eyvec[iiji]*ww05+eyvec[jiji]*ww06
            +eyvec[jjji]*ww07+eyvec[ijji]*ww08+eyvec[iiij]*ww09
            +eyvec[jiij]*ww10+eyvec[jjij]*ww11+eyvec[ijij]*ww12
            +eyvec[iijj]*ww13+eyvec[jijj]*ww14+eyvec[jjjj]*ww15
            +eyvec[ijjj]*ww16
        )
        
        ez[idx]=(
            ezvec[iiii]*ww01+ezvec[jiii]*ww02+ezvec[jjii]*ww03
            +ezvec[ijii]*ww04+ezvec[iiji]*ww05+ezvec[jiji]*ww06
            +ezvec[jjji]*ww07+ezvec[ijji]*ww08+ezvec[iiij]*ww09
            +ezvec[jiij]*ww10+ezvec[jjij]*ww11+ezvec[ijij]*ww12
            +ezvec[iijj]*ww13+ezvec[jijj]*ww14+ezvec[jjjj]*ww15
            +ezvec[ijjj]*ww16
        )
        
        #...calculate btot and field derivatives to 1st order...
        # ...first form more intermediate weights...
        w2m3m4m=w2m*w3m*w4m
        w23m4m=w2*w3m*w4m
        w2m34m=w2m*w3*w4m
        w234m=w2*w3*w4m
        w2m3m4=w2m*w3m*w4
        w23m4=w2*w3m*w4
        w2m34=w2m*w3*w4
        w234=w2*w3*w4
        
        w1m3m4m=w1m*w3m*w4m
        w13m4m=w1*w3m*w4m
        w1m34m=w1m*w3*w4m
        w134m=w1*w3*w4m
        w1m3m4=w1m*w3m*w4
        w13m4=w1*w3m*w4
        w1m34=w1m*w3*w4
        w134=w1*w3*w4
    
        w1m2m4m=w1m2m*w4m
        w12m4m=w12m*w4m
        w1m24m=w1m2*w4m
        w124m=w12*w4m
        w1m2m4=w1m2m*w4
        w12m4=w12m*w4
        w1m24=w1m2*w4
        w124=w12*w4
        
        #...calculate component derivatives...
        dbxdx[idx]=(
            ((bxvec[jiii]-bxvec[iiii])*w2m3m4m
            +(bxvec[jjii]-bxvec[ijii])*w23m4m
            +(bxvec[jiji]-bxvec[iiji])*w2m34m
            +(bxvec[jjji]-bxvec[ijji])*w234m
            +(bxvec[jiij]-bxvec[iiij])*w2m3m4
            +(bxvec[jjij]-bxvec[ijij])*w23m4
            +(bxvec[jijj]-bxvec[iijj])*w2m34
            +(bxvec[jjjj]-bxvec[ijjj])*w234)/dx
            -bfac1*y[idx, 2]+bfac2*y[idx, 0]*y[idx, 0]*y[idx, 2]
        )
        
        dbxdy[idx]=(
            ((bxvec[ijii]-bxvec[iiii])*w1m3m4m
             +(bxvec[jjii]-bxvec[jiii])*w13m4m
             +(bxvec[ijji]-bxvec[iiji])*w1m34m
             +(bxvec[jjji]-bxvec[jiji])*w134m
             +(bxvec[ijij]-bxvec[iiij])*w1m3m4
             +(bxvec[jjij]-bxvec[jiij])*w13m4
             +(bxvec[ijjj]-bxvec[iijj])*w1m34
             +(bxvec[jjjj]-bxvec[jijj])*w134)/dy
            +bfac2*y[idx, 0]*y[idx, 1]*y[idx, 2]
        )
        
        dbxdz[idx]=(
            ((bxvec[iiji]-bxvec[iiii])*w1m2m4m
             +(bxvec[jiji]-bxvec[jiii])*w12m4m
             +(bxvec[ijji]-bxvec[ijii])*w1m24m
             +(bxvec[jjji]-bxvec[jjii])*w124m
             +(bxvec[iijj]-bxvec[iiij])*w1m2m4
             +(bxvec[jijj]-bxvec[jiij])*w12m4
             +(bxvec[ijjj]-bxvec[ijij])*w1m24
            +(bxvec[jjjj]-bxvec[jjij])*w124)/dz
            -bfac1*y[idx, 0]+bfac2*y[idx, 0]*y[idx, 2]*y[idx, 2]
        )
        
        dbydx[idx]=(
            ((byvec[jiii]-byvec[iiii])*w2m3m4m
             +(byvec[jjii]-byvec[ijii])*w23m4m
             +(byvec[jiji]-byvec[iiji])*w2m34m
             +(byvec[jjji]-byvec[ijji])*w234m
             +(byvec[jiij]-byvec[iiij])*w2m3m4
             +(byvec[jjij]-byvec[ijij])*w23m4
             +(byvec[jijj]-byvec[iijj])*w2m34
             +(byvec[jjjj]-byvec[ijjj])*w234)/dx
            +bfac2*y[idx, 1]*y[idx, 2]*y[idx, 0]
        )
        
        dbydy[idx]=(
            ((byvec[ijii]-byvec[iiii])*w1m3m4m
             +(byvec[jjii]-byvec[jiii])*w13m4m
             +(byvec[ijji]-byvec[iiji])*w1m34m
             +(byvec[jjji]-byvec[jiji])*w134m
             +(byvec[ijij]-byvec[iiij])*w1m3m4
             +(byvec[jjij]-byvec[jiij])*w13m4
             +(byvec[ijjj]-byvec[iijj])*w1m34
             +(byvec[jjjj]-byvec[jijj])*w134)/dy
            -bfac1*y[idx, 2]+bfac2*y[idx, 1]*y[idx, 1]*y[idx, 2]
        )
        
        dbydz[idx]=(
            ((byvec[iiji]-byvec[iiii])*w1m2m4m
             +(byvec[jiji]-byvec[jiii])*w12m4m
             +(byvec[ijji]-byvec[ijii])*w1m24m
             +(byvec[jjji]-byvec[jjii])*w124m
             +(byvec[iijj]-byvec[iiij])*w1m2m4
             +(byvec[jijj]-byvec[jiij])*w12m4
             +(byvec[ijjj]-byvec[ijij])*w1m24
             +(byvec[jjjj]-byvec[jjij])*w124)/dz
            -bfac1*y[idx, 1]+bfac2*y[idx, 2]*y[idx, 2]*y[idx, 1]
        )
        
        dbzdx[idx]=(
            ((bzvec[jiii]-bzvec[iiii])*w2m3m4m
             +(bzvec[jjii]-bzvec[ijii])*w23m4m
             +(bzvec[jiji]-bzvec[iiji])*w2m34m
             +(bzvec[jjji]-bzvec[ijji])*w234m
             +(bzvec[jiij]-bzvec[iiij])*w2m3m4
             +(bzvec[jjij]-bzvec[ijij])*w23m4
             +(bzvec[jijj]-bzvec[iijj])*w2m34
             +(bzvec[jjjj]-bzvec[ijjj])*w234)/dx
            -bfac1*y[idx, 0]+bfac2*y[idx, 0]*y[idx, 2]*y[idx, 2]
        )
        
        dbzdy[idx]=(
            ((bzvec[ijii]-bzvec[iiii])*w1m3m4m
             +(bzvec[jjii]-bzvec[jiii])*w13m4m
             +(bzvec[ijji]-bzvec[iiji])*w1m34m
             +(bzvec[jjji]-bzvec[jiji])*w134m
             +(bzvec[ijij]-bzvec[iiij])*w1m3m4
             +(bzvec[jjij]-bzvec[jiij])*w13m4
             +(bzvec[ijjj]-bzvec[iijj])*w1m34
             +(bzvec[jjjj]-bzvec[jijj])*w134)/dy
            -bfac1*y[idx, 1]+bfac2*y[idx, 1]*y[idx, 2]*y[idx, 2]
        )
        
        dbzdz[idx]=(
            ((bzvec[iiji]-bzvec[iiii])*w1m2m4m
             +(bzvec[jiji]-bzvec[jiii])*w12m4m
             +(bzvec[ijji]-bzvec[ijii])*w1m24m
             +(bzvec[jjji]-bzvec[jjii])*w124m
             +(bzvec[iijj]-bzvec[iiij])*w1m2m4
             +(bzvec[jijj]-bzvec[jiij])*w12m4
             +(bzvec[ijjj]-bzvec[ijij])*w1m24
             +(bzvec[jjjj]-bzvec[jjij])*w124)/dz
            -3.*bfac1*y[idx, 2]+bfac2*y[idx, 2]*y[idx, 2]*y[idx, 2]
        )
        
        #      ...calculate btot...
        b[idx]=(bx[idx]**2.+by[idx]**2.+bz[idx]**2.)**(0.5)
        
        # ...calculate derivatives of btot...
        dbdx[idx]=(bx[idx]*dbxdx[idx] + by[idx]*dbydx[idx] + bz[idx]*dbzdx[idx])/b[idx]
        dbdy[idx]=(bx[idx]*dbxdy[idx] + by[idx]*dbydy[idx] + bz[idx]*dbzdy[idx])/b[idx]
        dbdz[idx]=(bx[idx]*dbxdz[idx] + by[idx]*dbydz[idx] + bz[idx]*dbzdz[idx])/b[idx]
    

def do_step(k1, k2, k3, k4, k5, k6, y, h, t, rtol, t_final, field_model, stopped, stopping_conditions):       
    """Do a Runge-Kutta Step.

    Args
      k1-k6: K values for Runge-Kutta
      y: current state vector
      h: current vector of step sizes
      t: current vector of particle times
      rtol: relative tolerance
      t_final: final time (dimensionalized)
      field_model: instance of libgputrace.FieldModel
    Returns
      num_iterated: number of particles iterated
    """
    # Evaluate Stopping Conditions
    if stopping_conditions:
        for stop_cond in stopping_conditions:
            stopped |= stop_cond(y, t, field_model)

    # NaN h happens when one of the rhs evaluations for k1-k2
    # is out of bounds
    stopped |= cp.isnan(h) 

    # Call Kernel to do the rest of the work
    arr_size = k1.shape[0]
    block_size = 1024
    grid_size = int(math.ceil(arr_size / block_size))
    y_next = cp.zeros(y.shape)
    z_next = cp.zeros(y.shape)    
    rtol_arr = cp.zeros(arr_size) + rtol
    t_final_arr = cp.zeros(arr_size) + t_final
    r_inner = cp.zeros(arr_size) + field_model.axes.r_inner
    mask = cp.zeros(arr_size, dtype=bool)
    
    do_step_kernel[grid_size, block_size](
        arr_size, k1, k2, k3, k4, k5, k6, y, y_next, z_next, h, t,
        rtol_arr, t_final_arr, mask, stopped,
        field_model.axes.x, field_model.axes.x.size,
        field_model.axes.y, field_model.axes.y.size,
        field_model.axes.z, field_model.axes.z.size,
        field_model.axes.t, field_model.axes.t.size,
        r_inner,
    )
    
    num_iterated = mask.sum()

    return num_iterated


@jit.rawkernel()
def do_step_kernel(
        arr_size, k1, k2, k3, k4, k5, k6,
        y, y_next, z_next, h, t, rtol, t_final, mask, stopped,
        x_axis, nx, y_axis, ny, z_axis, nz, t_axis, nt, r_inner):
    """[CUPY KERNEL] Do a Runge-Kutta Step
    
    Calculates error, selectively steps, and adjusts step size 
    """
    idx = jit.blockDim.x * jit.blockIdx.x + jit.threadIdx.x
    R = RK45Coeffs
    nstate = 5
    
    if idx < arr_size:
        # Compute thte total error in the position and momentum
        err_total = 0.0
        ynorm = 0.0
        
        for i in range(nstate):
            y_next[idx, i] = (
                y[idx, i]
                + R.c1 * k1[idx, i]
                + R.c3 * k3[idx, i]
                + R.c4 * k4[idx, i]
                + R.c5 * k5[idx, i]
            )
            z_next[idx, i] = (
                y[idx, i]
                + R.d1 * k1[idx, i]
                + R.d3 * k3[idx, i]
                + R.d4 * k4[idx, i]
                + R.d5 * k5[idx, i]
                + R.d6 * k6[idx, i]
            )
            err_total += (y_next[idx, i] - z_next[idx, i])**2
            ynorm += y[idx, i]**2
            
        err_total = err_total**(0.5)
        ynorm = ynorm**(0.5)
        
        # Compute the error tolerance
        tolerance = rtol[idx] * ynorm
        scale = 0.84*(tolerance/err_total)**(1/4)

        # Does not exceed target integration
        stopped[idx] |= t[idx] > t_final[idx]
        
        # Within x,y,z axes bounds
        stopped[idx] |= y[idx, 0] < x_axis[0]
        stopped[idx] |= y[idx, 1] < y_axis[0]
        stopped[idx] |= y[idx, 2] < z_axis[0]
                
        stopped[idx] |= y[idx, 0] > x_axis[nx-1]
        stopped[idx] |= y[idx, 1] > y_axis[ny-1]
        stopped[idx] |= y[idx, 2] > z_axis[nz-1]                

        radius = (y[idx, 0]**2 + y[idx, 1]**2 + y[idx, 2]**2)**0.5
        stopped[idx] |= radius < r_inner[idx]
        
        # WIthin t axes bounds
        stopped[idx] |= t[idx] < t_axis[0]
        stopped[idx] |= t[idx] > t_axis[nt-1]

        # Mask for iteration
        mask[idx] = (err_total < rtol[idx] * ynorm) & ~stopped[idx]
        
        # Selectively step particles
        if mask[idx]:
            t[idx] += h[idx]
            
            for i in range(nstate):
                y[idx, i] = z_next[idx, i] 

        h[idx] *= scale

                
def redim_time(val):
    """Redimensionalize a time value.

    Args
      value: value with units
    Returns
      value in redimensionalized units
    """
    sf = constants.c / constants.R_earth
    return (sf * val).to(1).value

                
def undim_time(val):
    """Redimensionalize a time value.

    Args
      value: value in seconds
    Returns
      value in redimensionalized units
    """
    sf = constants.R_earth / constants.c
    return val * sf

