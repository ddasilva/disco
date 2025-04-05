import cupy as cp
import sys
from cupyx import jit
import math
from dataclasses import dataclass
from typing import Any
from astropy import constants, units
from astropy.units import Quantity
from line_profiler import profile
import numpy as np


@dataclass
class TraceConfig:
    """Configuration for running the tracing code"""
    t_final: Quantity                    # end time of integration
    t_initial: Quantity = 0 * units.s    # start time of integration
    h_initial: Quantity = 1 * units.ms   # initial step size
    h_min: float = .1 * units.ms         # min step size
    h_max: float = 1 * units.s           # max step size
    rtol: float = 1e-3                   # relative tolerance
    grad_step: float = .1                # finite diff delta step (half)
    output_freq: int = 5                 # in iterations


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

    def __init__(self, Bx, By, Bz, Ex, Ey, Ez, mass, charge, B0=DEFAULT_RAW_B0):
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

    def multi_interp(self, t, y):
        """Abstract base method to be overridden."""
        raise NotImplementedError("This method should be defined in a subclass")

        

class UnstructuredFieldModel(FieldModel):
    """Set of magnetic and electric field models on unstructured grid.

    Interpolation is done by distance-weighted-averaging points within
    a neighborhood.
    """
    
    def __init__(self, Bx, By, Bz, Ex, Ey, Ez, mass, charge, point_cloud):
        """Get an instance that is dimensionalized and stored on the GPU.

        Notes
        - mass is not part of field model, but is used to redimensionalize.
        - input argument should have astropy units attached.
        - point_cloud is instance of UnstructuredFieldModelPointCloud
        """
        self.point_cloud = point_cloud
        super().__init__(Bx, By, Bz, Ex, Ey, Ez, mass, charge)
    
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
        raise NotImplementedError("TODO")


class RectilinearFieldModel(FieldModel):
    """Set of magnetic and electric field models on rectilinear grid

    See also:
      RectilinearAxes
    """
    def __init__(self, Bx, By, Bz, Ex, Ey, Ez, mass, charge, axes):
        """Get an instance that is dimensionalized and stored on the GPU.

        Notes
        - mass is not part of field model, but is used to redimensionalize.
        - input argument should have astropy units attached.
        - axes is an instance of RectilinearAxes
        """
        self.axes = axes
        super().__init__(Bx, By, Bz, Ex, Ey, Ez, mass, charge)


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
        
        xgr = self.axes.x
        ygr = self.axes.y
        zgr = self.axes.z
        tgr = self.axes.t
        
        ix, iy, iz, it = (
            neighbors.field_i,
            neighbors.field_j,
            neighbors.field_k,
            neighbors.field_l,            
        )

        bxdv = self.Bx.reshape(nttl, order='F')
        bydv = self.By.reshape(nttl, order='F')
        bzdv = self.Bz.reshape(nttl, order='F')
        exdv = self.Ex.reshape(nttl, order='F')
        eydv = self.Ey.reshape(nttl, order='F')
        ezdv = self.Ez.reshape(nttl, order='F')

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

        linterp4[grid_size, block_size](
            arr_size, nx, ny, nz, nxy, nxyz, nttl,
            ix, iy, iz, it,
            t, y, b0,
            tgr, xgr, ygr, zgr, 
            bxdv, bydv, bzdv, exdv, eydv, ezdv,
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


class UnstructuredFieldModelPointCloud:
    """Point cloud positions for unstructured field model.

    Attributes
      t: time axis
      x: x axis
      y: y axis
      z: z axis
    """
    DEFAULT_NEIGHBORHOOD_SIZE = 0.25
    
    def __init__(self, t, x, y, z, neighborhood_size=DEFAULT_NEIGHBORHOOD_SIZE):
        """Initialize instance that is dimensionalized and stored
        on the GPU.

        Input arguments should have astropy units
        """

        assert len(t.shape) == 1
        assert len(x.shape) == 2
        assert len(y.shape) == 2
        assert len(z.shape) == 2
                        
        Re = constants.R_earth
        self.t = cp.array(redim_time(t))
        self.x = cp.array((x / Re).to(1).value)
        self.y = cp.array((y / Re).to(1).value)
        self.z = cp.array((z / Re).to(1).value)

    def get_neighbors(self, t, pos_x, pos_y, pos_z):
        raise NotImplementedError()
        
        
class RectilinearAxes:
    """1D arrays of rectilinear grid axes

    Attributes
      t: time axis
      x: x axis
      y: y axis
      z: z axis
    """
    def __init__(self, t, x, y, z):
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
    

@profile
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
        if iter_count % config.output_freq == 0:
            gamma = cp.sqrt(1 + 2 * B * y[:, 4] + y[:, 3]**2)
            W = gamma - 1            
            hist_t.append(t.get())
            hist_y.append(y.get())
            hist_B.append(B.get())
            hist_W.append(W.get())
            hist_h.append(h.get())

        num_iterated = do_step(k1, k2, k3, k4, k5, k6, y, h, t, config.rtol, t_final)
        all_complete = cp.all(t >= t_final)
        iter_count += 1

        r_mean = cp.sqrt(y[:, 0]**2 + y[:, 1]**2 + y[:, 2]**2).mean()
        h_step = undim_time(float(h.mean())).to(units.ms).value
        
        print(f'Complete: {100 * min(t.min() / t_final, 1):.1f}% '
              f'(iter {iter_count}, {num_iterated} iterated, h mean '
              f'{h_step:.2f} ms, r mean {r_mean:.2f})')
        
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
    
    dydt = cp.zeros((arr_size, 5))

    rhs_kernel[grid_size, block_size](
        arr_size, y, dydt, 
        Bx, By, Bz, B,
        Ex, Ey, Ez,
        dBdx, dBdy, dBdz,
        dBxdy, dBxdz, dBydx, dBydz, dBzdx, dBzdy,
    )

    return dydt, B


@jit.rawkernel()
def rhs_kernel(
        arr_size, y, dydt_arr, 
        Bx_arr, By_arr, Bz_arr, B_arr,
        Ex_arr, Ey_arr, Ez_arr,
        dBdx_arr, dBdy_arr, dBdz_arr,
        dBxdy_arr, dBxdz_arr, dBydx_arr, dBydz_arr, dBzdx_arr, dBzdy_arr,
):
    """[CUPY KERNEL] implements RHS of oDE. 

    Code adapted from Fortran. Uses gyro-averaged equations of motion developed 
    by Brizzard and Chan (Phys. Plasmas 6, 4553, 1999),

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
def linterp4(
        arr_size, nx, ny, nz, nxy, nxyz, nttl,
        ix, iy, iz, it,
        t, y, b0,
        tgr, xgr, ygr, zgr, 
        bxdv, bydv, bzdv, exdv, eydv, ezdv,
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
    
    if idx < arr_size:
        dx = xgr[ix[idx] + 1] - xgr[ix[idx]]
        dy = ygr[iy[idx] + 1] - ygr[iy[idx]]
        dz = zgr[iz[idx] + 1] - zgr[iz[idx]]
        dt = tgr[it[idx] + 1] - tgr[it[idx]]

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
        w1=cp.abs(y[idx, 0] - xgr[ix[idx]])/dx
        w1m=1.-w1
        w2=cp.abs(y[idx, 1] - ygr[iy[idx]])/dy
        w2m=1.-w2
        w3=cp.abs(y[idx, 2] - zgr[iz[idx]])/dz
        w3m=1.-w3
        w4=cp.abs(t[idx] - tgr[it[idx]])/dt
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
            bxdv[iiii]*ww01+bxdv[jiii]*ww02+bxdv[jjii]*ww03
            +bxdv[ijii]*ww04+bxdv[iiji]*ww05+bxdv[jiji]*ww06
            +bxdv[jjji]*ww07+bxdv[ijji]*ww08+bxdv[iiij]*ww09
            +bxdv[jiij]*ww10+bxdv[jjij]*ww11+bxdv[ijij]*ww12
            +bxdv[iijj]*ww13+bxdv[jijj]*ww14+bxdv[jjjj]*ww15
            +bxdv[ijjj]*ww16 
            -bfac1*y[idx, 0]*y[idx, 2]
        )

        by[idx]=(
            bydv[iiii]*ww01+bydv[jiii]*ww02+bydv[jjii]*ww03
            +bydv[ijii]*ww04+bydv[iiji]*ww05+bydv[jiji]*ww06
            +bydv[jjji]*ww07+bydv[ijji]*ww08+bydv[iiij]*ww09
            +bydv[jiij]*ww10+bydv[jjij]*ww11+bydv[ijij]*ww12
            +bydv[iijj]*ww13+bydv[jijj]*ww14+bydv[jjjj]*ww15
            +bydv[ijjj]*ww16 
            -bfac1*y[idx, 1]*y[idx, 2]
        )
        
        bz[idx]=(
            bzdv[iiii]*ww01+bzdv[jiii]*ww02+bzdv[jjii]*ww03
            +bzdv[ijii]*ww04+bzdv[iiji]*ww05+bzdv[jiji]*ww06
            +bzdv[jjji]*ww07+bzdv[ijji]*ww08+bzdv[iiij]*ww09
            +bzdv[jiij]*ww10+bzdv[jjij]*ww11+bzdv[ijij]*ww12
            +bzdv[iijj]*ww13+bzdv[jijj]*ww14+bzdv[jjjj]*ww15
            +bzdv[ijjj]*ww16 
            -bfac1*y[idx, 2]*y[idx, 2] + b0[idx]/r2/r
        )

        ex[idx]=(
            exdv[iiii]*ww01+exdv[jiii]*ww02+exdv[jjii]*ww03
            +exdv[ijii]*ww04+exdv[iiji]*ww05+exdv[jiji]*ww06
            +exdv[jjji]*ww07+exdv[ijji]*ww08+exdv[iiij]*ww09
            +exdv[jiij]*ww10+exdv[jjij]*ww11+exdv[ijij]*ww12
            +exdv[iijj]*ww13+exdv[jijj]*ww14+exdv[jjjj]*ww15
            +exdv[ijjj]*ww16
        )
        
        ey[idx]=(
            eydv[iiii]*ww01+eydv[jiii]*ww02+eydv[jjii]*ww03
            +eydv[ijii]*ww04+eydv[iiji]*ww05+eydv[jiji]*ww06
            +eydv[jjji]*ww07+eydv[ijji]*ww08+eydv[iiij]*ww09
            +eydv[jiij]*ww10+eydv[jjij]*ww11+eydv[ijij]*ww12
            +eydv[iijj]*ww13+eydv[jijj]*ww14+eydv[jjjj]*ww15
            +eydv[ijjj]*ww16
        )
        
        ez[idx]=(
            ezdv[iiii]*ww01+ezdv[jiii]*ww02+ezdv[jjii]*ww03
            +ezdv[ijii]*ww04+ezdv[iiji]*ww05+ezdv[jiji]*ww06
            +ezdv[jjji]*ww07+ezdv[ijji]*ww08+ezdv[iiij]*ww09
            +ezdv[jiij]*ww10+ezdv[jjij]*ww11+ezdv[ijij]*ww12
            +ezdv[iijj]*ww13+ezdv[jijj]*ww14+ezdv[jjjj]*ww15
            +ezdv[ijjj]*ww16
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
            ((bxdv[jiii]-bxdv[iiii])*w2m3m4m
            +(bxdv[jjii]-bxdv[ijii])*w23m4m
            +(bxdv[jiji]-bxdv[iiji])*w2m34m
            +(bxdv[jjji]-bxdv[ijji])*w234m
            +(bxdv[jiij]-bxdv[iiij])*w2m3m4
            +(bxdv[jjij]-bxdv[ijij])*w23m4
            +(bxdv[jijj]-bxdv[iijj])*w2m34
            +(bxdv[jjjj]-bxdv[ijjj])*w234)/dx
            -bfac1*y[idx, 2]+bfac2*y[idx, 0]*y[idx, 0]*y[idx, 2]
        )
        
        dbxdy[idx]=(
            ((bxdv[ijii]-bxdv[iiii])*w1m3m4m
             +(bxdv[jjii]-bxdv[jiii])*w13m4m
             +(bxdv[ijji]-bxdv[iiji])*w1m34m
             +(bxdv[jjji]-bxdv[jiji])*w134m
             +(bxdv[ijij]-bxdv[iiij])*w1m3m4
             +(bxdv[jjij]-bxdv[jiij])*w13m4
             +(bxdv[ijjj]-bxdv[iijj])*w1m34
             +(bxdv[jjjj]-bxdv[jijj])*w134)/dy
            +bfac2*y[idx, 0]*y[idx, 1]*y[idx, 2]
        )
        
        dbxdz[idx]=(
            ((bxdv[iiji]-bxdv[iiii])*w1m2m4m
             +(bxdv[jiji]-bxdv[jiii])*w12m4m
             +(bxdv[ijji]-bxdv[ijii])*w1m24m
             +(bxdv[jjji]-bxdv[jjii])*w124m
             +(bxdv[iijj]-bxdv[iiij])*w1m2m4
             +(bxdv[jijj]-bxdv[jiij])*w12m4
             +(bxdv[ijjj]-bxdv[ijij])*w1m24
            +(bxdv[jjjj]-bxdv[jjij])*w124)/dz
            -bfac1*y[idx, 0]+bfac2*y[idx, 0]*y[idx, 2]*y[idx, 2]
        )
        
        dbydx[idx]=(
            ((bydv[jiii]-bydv[iiii])*w2m3m4m
             +(bydv[jjii]-bydv[ijii])*w23m4m
             +(bydv[jiji]-bydv[iiji])*w2m34m
             +(bydv[jjji]-bydv[ijji])*w234m
             +(bydv[jiij]-bydv[iiij])*w2m3m4
             +(bydv[jjij]-bydv[ijij])*w23m4
             +(bydv[jijj]-bydv[iijj])*w2m34
             +(bydv[jjjj]-bydv[ijjj])*w234)/dx
            +bfac2*y[idx, 1]*y[idx, 2]*y[idx, 0]
        )
        
        dbydy[idx]=(
            ((bydv[ijii]-bydv[iiii])*w1m3m4m
             +(bydv[jjii]-bydv[jiii])*w13m4m
             +(bydv[ijji]-bydv[iiji])*w1m34m
             +(bydv[jjji]-bydv[jiji])*w134m
             +(bydv[ijij]-bydv[iiij])*w1m3m4
             +(bydv[jjij]-bydv[jiij])*w13m4
             +(bydv[ijjj]-bydv[iijj])*w1m34
             +(bydv[jjjj]-bydv[jijj])*w134)/dy
            -bfac1*y[idx, 2]+bfac2*y[idx, 1]*y[idx, 1]*y[idx, 2]
        )
        
        dbydz[idx]=(
            ((bydv[iiji]-bydv[iiii])*w1m2m4m
             +(bydv[jiji]-bydv[jiii])*w12m4m
             +(bydv[ijji]-bydv[ijii])*w1m24m
             +(bydv[jjji]-bydv[jjii])*w124m
             +(bydv[iijj]-bydv[iiij])*w1m2m4
             +(bydv[jijj]-bydv[jiij])*w12m4
             +(bydv[ijjj]-bydv[ijij])*w1m24
             +(bydv[jjjj]-bydv[jjij])*w124)/dz
            -bfac1*y[idx, 1]+bfac2*y[idx, 2]*y[idx, 2]*y[idx, 1]
        )
        
        dbzdx[idx]=(
            ((bzdv[jiii]-bzdv[iiii])*w2m3m4m
             +(bzdv[jjii]-bzdv[ijii])*w23m4m
             +(bzdv[jiji]-bzdv[iiji])*w2m34m
             +(bzdv[jjji]-bzdv[ijji])*w234m
             +(bzdv[jiij]-bzdv[iiij])*w2m3m4
             +(bzdv[jjij]-bzdv[ijij])*w23m4
             +(bzdv[jijj]-bzdv[iijj])*w2m34
             +(bzdv[jjjj]-bzdv[ijjj])*w234)/dx
            -bfac1*y[idx, 0]+bfac2*y[idx, 0]*y[idx, 2]*y[idx, 2]
        )
        
        dbzdy[idx]=(
            ((bzdv[ijii]-bzdv[iiii])*w1m3m4m
             +(bzdv[jjii]-bzdv[jiii])*w13m4m
             +(bzdv[ijji]-bzdv[iiji])*w1m34m
             +(bzdv[jjji]-bzdv[jiji])*w134m
             +(bzdv[ijij]-bzdv[iiij])*w1m3m4
             +(bzdv[jjij]-bzdv[jiij])*w13m4
             +(bzdv[ijjj]-bzdv[iijj])*w1m34
             +(bzdv[jjjj]-bzdv[jijj])*w134)/dy
            -bfac1*y[idx, 1]+bfac2*y[idx, 1]*y[idx, 2]*y[idx, 2]
        )
        
        dbzdz[idx]=(
            ((bzdv[iiji]-bzdv[iiii])*w1m2m4m
             +(bzdv[jiji]-bzdv[jiii])*w12m4m
             +(bzdv[ijji]-bzdv[ijii])*w1m24m
             +(bzdv[jjji]-bzdv[jjii])*w124m
             +(bzdv[iijj]-bzdv[iiij])*w1m2m4
             +(bzdv[jijj]-bzdv[jiij])*w12m4
             +(bzdv[ijjj]-bzdv[ijij])*w1m24
             +(bzdv[jjjj]-bzdv[jjij])*w124)/dz
            -3.*bfac1*y[idx, 2]+bfac2*y[idx, 2]*y[idx, 2]*y[idx, 2]
        )
        
        #      ...calculate btot...
        b[idx]=(bx[idx]**2.+by[idx]**2.+bz[idx]**2.)**(0.5)
        
        # ...calculate derivatives of btot...
        dbdx[idx]=(bx[idx]*dbxdx[idx] + by[idx]*dbydx[idx] + bz[idx]*dbzdx[idx])/b[idx]
        dbdy[idx]=(bx[idx]*dbxdy[idx] + by[idx]*dbydy[idx] + bz[idx]*dbzdy[idx])/b[idx]
        dbdz[idx]=(bx[idx]*dbxdz[idx] + by[idx]*dbydz[idx] + bz[idx]*dbzdz[idx])/b[idx]
    

@profile
def do_step(k1, k2, k3, k4, k5, k6, y, h, t, rtol, t_final):       
    """Do a Runge-Kutta Step.

    Args
      k1-k6: K values for Runge-Kutta
      y: current state vector
      h: current vector of step sizes
      t: current vector of particle times
      rtol: relative tolerance
      t_final: final time (dimensionalized)
    Returns
      num_iterated: number of particles iterated
    """
    arr_size = k1.shape[0]
    block_size = 1024
    grid_size = int(math.ceil(arr_size / block_size))
    y_next = cp.zeros(y.shape)
    z_next = cp.zeros(y.shape)    
    rtol_arr = cp.zeros(arr_size) + rtol
    t_final_arr = cp.zeros(arr_size) + t_final
    mask = cp.zeros(arr_size, dtype=bool)
        
    do_step_kernel[grid_size, block_size](
        arr_size, k1, k2, k3, k4, k5, k6, y, y_next, z_next, h, t,
        rtol_arr, t_final_arr, mask
    )
    
    num_iterated = mask.sum()

    return num_iterated


@jit.rawkernel()
def do_step_kernel(
        arr_size, k1, k2, k3, k4, k5, k6,
        y, y_next, z_next, h, t, rtol, t_final, mask):
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
        mask[idx] = (err_total < rtol[idx] * ynorm) & (t[idx] < t_final[idx])
        
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

