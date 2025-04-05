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
    h_initial: Quantity = 5 * units.ms   # initial step size
    h_min: float = .1 * units.ms         # min step size
    h_max: float = 1 * units.s           # max step size
    rtol: float = 1e-3                   # relative tolerance
    grad_step: float = 5e-2              # finite diff delta step (half)
    output_freq: int = 10                # in iterations


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
    DEFAULT_RAW_B0 = -31e3 * units.nT

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

    def interp(self, field, t, pos_x, pos_y, pos_z, axes, neighbors=None):
        """Abstract base method to be overridden."""
        raise NotImplementedError("This method should be defined in a subclass")

    def get_dipole_table(self, y, grad_step):
        """Get a table of precomputed dipole values for finite 
        differencing in RHS.

        Args
           y: Differential equation state vector
           grad_step: finite difference step to generate table with        
        Returns
           A dictionary. Keys are like:
        
           # at position
           dipole_table['Bx'] 
        
           # forward in the dx direction
           dipole_table['Bx', 'forward', 'dx'] 

           # backwards in the dy direction
           dipole_table['Bx', 'backwards', 'dy'] 
        """
        # Launch Kernel to handle rest 
        arr_size = y.shape[0]
        block_size = 1024
        grid_size = int(math.ceil(arr_size / block_size))
        
        B0s = cp.zeros(arr_size) + self.B0
        eps = cp.zeros(arr_size) + grad_step
        dipole_table = cp.zeros((arr_size, 3, 4, 2))
        
        dipole_table_kernel[grid_size, block_size](
            arr_size, y, eps, B0s, dipole_table,
        )

        # Convert to dictionary to make code more readable
        # when using it
        dipole_table_dict = {}

        for v, variable in enumerate(['Bx', 'By', 'Bz']):
            for i, step in enumerate(['dx', 'dy', 'dz', 'unchanged']):
                for j, direction in enumerate(['forward', 'backward']):
                    if step == 'unchanged':
                        dipole_table_dict[variable, step] = (
                            dipole_table[:, v, i, j]
                        )
                    else:
                        dipole_table_dict[variable, direction, step] = (
                            dipole_table[:, v, i, j]
                        )

        return dipole_table_dict
                    
    def get_dipole(self, pos_x, pos_y, pos_z, values=("Bx", "By", "Bz")):
        """Get dipole values at positions.
        
        Args
          pos_x: x position vector
          pos_y: y position ector
          pos_z: z position vector
          values: what dipole values to request
        Returns
           if only one value requested, returns that value
           otherwise returns a tuple of values
        """
        pos_r = np.sqrt(pos_x**2 + pos_y**2 + pos_z**2)
        rv = []

        if 'Bx' in values:    
            rv.append(3 * pos_x * pos_z * self.B0 / pos_r**5)
        if 'By' in values:
            rv.append(3 * pos_y * pos_z * self.B0 / pos_r**5)
        if 'Bz' in values:
            rv.append((3 * pos_z**2 - pos_r**2) * self.B0 / pos_r**5)

        if len(rv) == 1:
            return rv[0]
        else:            
            return rv
        

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
    
    def interp(self, field, t, pos_x, pos_y, pos_z, neighbors=None):
        """Interpolate field at given positions.
        
        Args
          field: field variable to interpolate
          t: time to interpolate at
          pos_x, pos_y, pos_z: particle positions
          neighbors: optional, reuse this value for lookup of the neighbors
          dx, dy, dz: optional, perturb the position by these values    
        Return
          result: cupy array of interpolated field values at position
          neighbors: neighbors object for reuse
        """     
        

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
    
    def interp(self, field, t, pos_x, pos_y, pos_z, neighbors=None):
        """Interpolate a 3D gridded field at given positions.
        
        Args
          field: 3D gridded field to interpolate
          t: time to interpolate at
          pos_x, pos_y, pos_z: particle positions
          neighbors: optional, reuse this value for lookup of the neighbors
          dx, dy, dz: optional, perturb the position by these values    
        Return
          result: cupy array of interpolated field values at position
          neighbors: neighbors object for reuse
        """     
        if neighbors is None:
            neighbors = self.axes.get_neighbors(t, pos_x, pos_y, pos_z)

        result = cp.zeros(pos_x.shape)

        arr_size = pos_x.size
        block_size = 256
        grid_size = int(math.ceil(arr_size / block_size))
        
        interp_quadlinear_kernel[grid_size, block_size](
            arr_size, result, field,
            neighbors.field_i,
            neighbors.field_j,
            neighbors.field_k,
            neighbors.field_l,
            t, pos_x, pos_y, pos_z,
            self.axes.t, self.axes.x, self.axes.y, self.axes.z
        )

        return result, neighbors
    

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
            field_i=cp.searchsorted(self.t, t, side='right'),
            field_j=cp.searchsorted(self.x, pos_x),
            field_k=cp.searchsorted(self.y, pos_y),
            field_l=cp.searchsorted(self.z, pos_z),
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
    t = cp.zeros(particle_state.x.size)
    t[:] = redim_time(config.t_initial)
    
    y = cp.zeros((particle_state.x.size, 5))
    y[:, 0] = particle_state.x
    y[:, 1] = particle_state.y
    y[:, 2] = particle_state.z
    y[:, 3] = particle_state.ppar
    y[:, 4] = particle_state.magnetic_moment

    h = cp.ones(particle_state.x.size) * (
        redim_time(config.h_initial)
    )
    h_min = cp.ones(particle_state.x.size) * (
        redim_time(config.h_min)
    )
    h_max = cp.ones(particle_state.x.size) * (
        redim_time(config.h_max)
    )

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
            field_model, config
        )
        k2 = h_ * rhs(
            t + h * R.a2,
            y + h_ * R.b21 * k1, field_model, config
        )
        k3 = h_ * rhs(
            t + h * R.a3,
            y + h_ * (R.b31*k1 + R.b32*k2), field_model, config
        )
        k4 = h_ * rhs(
            t + h * R.a4,
            y + h_ * (R.b41 * k1 + R.b42 * k2 + R.b43 * k3),
            field_model, config
        )
        k5 = h_ * rhs(
            t + h,
            y + h_ * (R.b51 * k1 + R.b52 * k2 + R.b53 * k3 + R.b54 * k4),
            field_model, config
        )
        k6 = h_ * rhs(
            t + h * R.a6,
            y + h_ * (R.b61 * k1 + R.b62 * k2 + R.b63 * k3 + R.b64 * k4 + R.b65 * k5),
            field_model, config
        )

        num_iterated = do_step(k1, k2, k3, k4, k5, k6, y, h, t, config.rtol, t_final)

        all_complete = cp.all(t >= t_final)
        iter_count += 1

        # Save incremented particles to history
        if iter_count % config.output_freq == 0:
            tmp_Bx, _ = field_model.interp(
                field_model.Bx, t, y[:, 0], y[:, 1], y[:, 2]
            )
            tmp_By, _ = field_model.interp(
                field_model.By, t, y[:, 0], y[:, 1], y[:, 2]
            )
            tmp_Bz, _ = field_model.interp(
                field_model.Bz, t, y[:, 0], y[:, 1], y[:, 2]
            )
            Bx_dip, By_dip, Bz_dip = field_model.get_dipole(y[:, 0], y[:, 1], y[:, 2])
            tmp_B = cp.sqrt((tmp_Bx + Bx_dip)**2 + (tmp_By + By_dip)**2 + (tmp_Bz + Bz_dip)**2)
            if field_model.negative_charge:
                tmp_B *= -1
                
            tmp_gamma = cp.sqrt(
                1 + 2 * tmp_B * y[:, 4] + y[:, 3]**2
            )
            tmp_W = tmp_gamma - 1
            
            hist_t.append(t.get())
            hist_y.append(y.get())
            hist_B.append(tmp_B.get())
            hist_W.append(tmp_W.get())
            hist_h.append(h.get())

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
    pos_x = y[:, 0]
    pos_y = y[:, 1]
    pos_z = y[:, 2]             
    dipole_table = field_model.get_dipole_table(y, config.grad_step)
    
    # Get B and E at particle position
    Bx, neighbors = field_model.interp(
        field_model.Bx, t, pos_x, pos_y, pos_z,
    )
    By, _ = field_model.interp(
        field_model.By, t, pos_x, pos_y, pos_z, neighbors=neighbors
    )
    Bz, _ = field_model.interp(
        field_model.Bz, t, pos_x, pos_y, pos_z, neighbors=neighbors
    )

    Bx += dipole_table['Bx', 'unchanged']
    By += dipole_table['By', 'unchanged']
    Bz += dipole_table['Bz', 'unchanged']
    
    Ex, _ = field_model.interp(
        field_model.Ex, t, pos_x, pos_y, pos_z, neighbors=neighbors
    )
    Ey, _ = field_model.interp(
        field_model.Ey, t, pos_x, pos_y, pos_z, neighbors=neighbors
    )
    Ez, _ = field_model.interp(
        field_model.Ez, t, pos_x, pos_y, pos_z, neighbors=neighbors
    )
    
    # Get derivatives from finite difference
    # ---------------------------------------
    eps = config.grad_step
    eps2 = 2 * eps
    
    pos_x_forw = pos_x + eps
    pos_y_forw = pos_y + eps
    pos_z_forw = pos_z + eps
    
    pos_x_back = pos_x - eps
    pos_y_back = pos_y - eps
    pos_z_back = pos_z - eps
    
    # dBx/dx
    dBxdx_forw, dx_forw_neighbors = field_model.interp(
        field_model.Bx, t, pos_x_forw, pos_y, pos_z,
    )
    dBxdx_back, dx_back_neighbors = field_model.interp(
        field_model.Bx, t, pos_x_back, pos_y, pos_z,
    )
    dBxdx_forw += dipole_table["Bx", "forward", "dx"]
    dBxdx_back += dipole_table["Bx", "backward", "dx"]
    dBxdx = (dBxdx_forw - dBxdx_back) / eps2
    
    # dBx/dy
    dBxdy_forw, dy_forw_neighbors = field_model.interp(
        field_model.Bx, t, pos_x, pos_y_forw, pos_z,
    )
    dBxdy_back, dy_back_neighbors = field_model.interp(
        field_model.Bx, t, pos_x, pos_y_back, pos_z,
    )
    dBxdy_forw += dipole_table["Bx", "forward", "dy"]
    dBxdy_back += dipole_table["Bx", "backward", "dy"]
    dBxdy = (dBxdy_forw - dBxdy_back) / eps2

    # dBx/dz
    dBxdz_forw, dz_forw_neighbors = field_model.interp(
        field_model.Bx, t, pos_x, pos_y, pos_z_forw,
    )
    dBxdz_back, dz_back_neighbors = field_model.interp(
        field_model.Bx, t, pos_x, pos_y, pos_z_back,
    )
    dBxdz_forw += dipole_table["Bx", "forward", "dz"]
    dBxdz_back += dipole_table["Bx", "backward", "dz"]
    dBxdz = (dBxdz_forw - dBxdz_back) / eps2
    
    # dBy/dx
    dBydx_forw, _ = field_model.interp(
        field_model.By, t, pos_x_forw, pos_y, pos_z,
        neighbors=dx_forw_neighbors
    )
    dBydx_back, _ = field_model.interp(
        field_model.By, t, pos_x_back, pos_y, pos_z,
        neighbors=dx_back_neighbors
    )    
    dBydx_forw += dipole_table["By", "forward", "dx"]
    dBydx_back += dipole_table["By", "backward", "dx"]
    dBydx = (dBydx_forw - dBydx_back) / eps2

    # dBy/dy
    dBydy_forw, _ = field_model.interp(
        field_model.By, t, pos_x, pos_y_forw, pos_z,
        neighbors=dy_forw_neighbors
    )
    dBydy_back, _ = field_model.interp(
        field_model.By, t, pos_x, pos_y_back, pos_z,
        neighbors=dy_back_neighbors
    )    
    dBydy_forw += dipole_table["By", "forward", "dy"]
    dBydy_back += dipole_table["By", "backward", "dy"]

    dBydy = (dBydy_forw - dBydy_back) / eps2
    
    # dBy/dz
    dBydz_forw, _ = field_model.interp(
        field_model.By, t, pos_x, pos_y, pos_z_forw,
        neighbors=dz_forw_neighbors
    )
    dBydz_back, _ = field_model.interp(
        field_model.By, t, pos_x, pos_y, pos_z_back, 
        neighbors=dz_back_neighbors
    )
    dBydz_forw += dipole_table["By", "forward", "dz"]
    dBydz_back += dipole_table["By", "backward", "dz"]
    dBydz = (dBydz_forw - dBydz_back) / eps2

    # dBz/dx
    dBzdx_forw, _ = field_model.interp(
        field_model.Bz, t, pos_x_forw, pos_y, pos_z, 
        neighbors=dx_forw_neighbors
    )
    dBzdx_back, _ = field_model.interp(
        field_model.Bz, t, pos_x_back, pos_y, pos_z, 
        neighbors=dx_back_neighbors
    )
    dBzdx_forw += dipole_table["Bz", "forward", "dx"]
    dBzdx_back += dipole_table["Bz", "backward", "dx"]
    dBzdx = (dBzdx_forw - dBzdx_back) / eps2

    # dBz/dy
    dBzdy_forw, _ = field_model.interp(
        field_model.Bz, t, pos_x, pos_y_forw, pos_z, 
        neighbors=dy_forw_neighbors
    )
    dBzdy_back, _ = field_model.interp(
        field_model.Bz, t, pos_x, pos_y_back, pos_z, 
        neighbors=dy_back_neighbors
    )
    #dBzdy_forw += field_model.get_dipole(pos_x, pos_y_forw, pos_z, "Bz")
    #dBzdy_back += field_model.get_dipole(pos_x, pos_y_back, pos_z, "Bz")    
    dBzdy_forw += dipole_table["Bz", "forward", "dy"]
    dBzdy_back += dipole_table["Bz", "backward", "dy"]

    dBzdy = (dBzdy_forw - dBzdy_back) / eps2

    # dBz/dz
    dBzdz_forw, _ = field_model.interp(
        field_model.Bz, t, pos_x, pos_y, pos_z_forw, 
        neighbors=dz_forw_neighbors
    )
    dBzdz_back, _ = field_model.interp(
        field_model.Bz, t, pos_x, pos_y, pos_z_back, 
        neighbors=dz_back_neighbors
    )
    #dBzdz_forw += field_model.get_dipole(pos_x, pos_y, pos_z_forw, "Bz")
    #dBzdz_back += field_model.get_dipole(pos_x, pos_y, pos_z_back, "Bz")    
    dBzdz_forw += dipole_table["Bz", "forward", "dz"]
    dBzdz_back += dipole_table["Bz", "backward", "dz"]

    dBzdz = (dBzdz_forw - dBzdz_back) / eps2
        
    # in |B| magnitude
    B = cp.sqrt(Bx**2 + By**2 + Bz**2)

    dBdx_forw = cp.sqrt(dBxdx_forw**2 + dBydx_forw**2 + dBzdx_forw**2)
    dBdx_back = cp.sqrt(dBxdx_back**2 + dBydx_back**2 + dBzdx_back**2)    
    dBdx = (dBdx_forw - dBdx_back) / eps2

    dBdy_forw = cp.sqrt(dBxdy_forw**2 + dBydy_forw**2 + dBzdy_forw**2)
    dBdy_back = cp.sqrt(dBxdy_back**2 + dBydy_back**2 + dBzdy_back**2)    
    dBdy = (dBdy_forw - dBdy_back) / eps2

    dBdz_forw = cp.sqrt(dBxdz_forw**2 + dBydz_forw**2 + dBzdz_forw**2)
    dBdz_back = cp.sqrt(dBxdz_back**2 + dBydz_back**2 + dBzdz_back**2)    
    dBdz = (dBdz_forw - dBdz_back) / eps2

    # need to account for dimensionalization of magnitude
    if field_model.negative_charge:
        B *= -1
        dBdx *= -1
        dBdy *= -1
        dBdz *= -1        
    
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


@jit.rawkernel()
def dipole_table_kernel(arr_size, y, eps, B0, dipole_table):
    """[CUPY KERNEL] Generates a dipole table for the RHS
    and finite differencing.
    """    
    idx = jit.blockDim.x * jit.blockIdx.x + jit.threadIdx.x
    
    if idx < arr_size:
        for i in range(4):
            for j in range(2):
                # Set flip direction
                if j == 0:
                    flip = 1
                else:
                    flip = -1

                # Get x,y,z coordinates
                if i == 0:
                    pos_x = y[idx, 0] + flip * eps[idx]
                else:
                    pos_x = y[idx, 0]
                    
                if i == 1:
                    pos_y = y[idx, 1] + flip * eps[idx]
                else:
                    pos_y = y[idx, 1]

                if i == 2:
                    pos_z = y[idx, 2] + flip * eps[idx]
                else:
                    pos_z = y[idx, 2]

                # Calculate dipole euqations
                pos_r = (pos_x*pos_x + pos_y*pos_y + pos_z*pos_z)**(1/2)
                
                Bx = 3 * pos_x * pos_z * B0[idx] / pos_r**5
                By = 3 * pos_y * pos_z * B0[idx] / pos_r**5
                Bz = (3 * pos_z*pos_z - pos_r*pos_r) * B0[idx] / pos_r**5

                # Store
                dipole_table[idx, 0, i, j] = Bx
                dipole_table[idx, 1, i, j] = By
                dipole_table[idx, 2, i, j] = Bz
    

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
                + R.c1*k1[idx, i]
                + R.c3*k3[idx, i]
                + R.c4*k4[idx, i]
                + R.c5*k5[idx, i]
            )
            z_next[idx, i] = (
                y[idx, i]
                + R.d1*k1[idx, i]
                + R.d3*k3[idx, i]
                + R.d4*k4[idx, i]
                + R.d5*k5[idx, i]
                + R.d6*k6[idx, i]
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
                y[idx, i] = y_next[idx, i] 

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


