import time
import numpy as np
import pandas as pd

import libgputrace 

from astropy import constants, units
from scipy.constants import elementary_charge


EARTH_DIPOLE_B0 = -30e3   # nT


def main():
    """Main method of the program."""
    config = libgputrace.TraceConfig(t_final=500)
    grid_spacing = 0.1 
    
    # Setup axes and grid
    x_axis = np.arange(-10, 10, grid_spacing) * units.R_earth
    y_axis = np.arange(-10, 10, grid_spacing) * units.R_earth
    z_axis = np.arange(-5, 5, grid_spacing) * units.R_earth
    
    x_grid, y_grid, z_grid = np.meshgrid(
        y_axis, y_axis, z_axis,
        indexing='ij'
    )
    axes = libgputrace.Axes.initialize(x_axis, y_axis, z_axis)
    
    print('Grid Shape:', x_grid.shape)
    
    # Instantiate particle state and parallel velocity
    pos_x = np.arange(3, 8, .01) * constants.R_earth
    pos_y = np.zeros(pos_x.shape) * constants.R_earth
    pos_z = np.zeros(pos_x.shape) * constants.R_earth
    vpar = np.ones(pos_x.shape) * 1e-2 * (constants.R_earth / units.s)
    magnetic_moment = np.ones(pos_x.shape) * 1e-36 * 1e9 * units.A * constants.R_earth**2
    charge = - elementary_charge * units.C
    mass = constants.m_e
    particle_state = libgputrace.ParticleState.initialize(
        pos_x, pos_y, pos_z, 
        vpar, magnetic_moment, mass, charge
    )
    
    print('Number of particles:', pos_x.size)

    # Calculate Dipole on Grid
    x_grid = (x_grid / constants.R_earth).to(1).value
    y_grid = (y_grid / constants.R_earth).to(1).value
    z_grid = (z_grid / constants.R_earth).to(1).value
    r_grid = np.sqrt(x_grid**2 + y_grid**2 + z_grid**2)
    
    Bx = 3 * x_grid * z_grid * EARTH_DIPOLE_B0 / r_grid**5
    By = 3 * y_grid * z_grid * EARTH_DIPOLE_B0 / r_grid**5
    Bz = (3 * z_grid**2 - r_grid**2) * EARTH_DIPOLE_B0 / r_grid**5
    B = np.sqrt(Bx**2 + By**2 + Bz**2)

    Bx = Bx * units.nT
    By = By * units.nT
    Bz = Bz * units.nT
    B *= units.nT
    Ex = np.zeros(Bx.shape) * units.mV/units.m
    Ey = np.zeros(Bx.shape) * units.mV/units.m
    Ez = np.zeros(Bx.shape) * units.mV/units.m

    field_model = libgputrace.FieldModel.initialize(
        Bx, By, Bz, B, Ex, Ey, Ez, mass, charge,
    )
    
    # History of positions
    #hist = libgputrace.ParticleHistory.initialize(config.max_iters, pos_x.size)

    #print('Iterations:', config.max_iters)
    
    start_time = time.time()
    libgputrace.trace_trajectory(config, particle_state, None, field_model, axes)
    end_time = time.time()
    
    print('time = ', end_time - start_time, 's')

    # Collect history
    hist_t = hist.t.get()
    hist_x = hist.x.get()
    hist_y = hist.y.get()
    hist_z = hist.z.get()
    hist_vpar = hist.vpar.get()
    hist_B = hist.B.get()
    hist_W = hist.W.get()
    
    # Write output for visualization
    d = {}
    
    for i in range(0, particle_state.x.size, 50):
        d[f't{i}'] = hist_t[:, i]
        d[f'x{i}'] = hist_x[:, i]
        d[f'y{i}'] = hist_y[:, i]
        d[f'z{i}'] = hist_z[:, i] 
        d[f'vpar{i}'] = hist_vpar[:, i] 
        d[f'B{i}'] = hist_B[:, i]
        d[f'W{i}'] = hist_W[:, i]
        
    pd.DataFrame(d).to_csv('data/test_dipole.csv')


if __name__ == '__main__':
    main()
