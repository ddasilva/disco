import time
import numpy as np
import pandas as pd

import libgputrace 

from astropy import constants, units
from scipy.constants import elementary_charge


EARTH_DIPOLE_B0 = -30e3   # nT


def main():
    """Main method of the program."""
    config = libgputrace.TraceConfig(t_final=1, h_initial=1e-2, h_min=1e-10, rtol=1e-3)
    grid_spacing = 0.1
    
    # Setup axes and grid
    t_axis = np.array([0, 100]) * units.s
    x_axis = np.arange(-10, 10, grid_spacing) * units.R_earth
    y_axis = np.arange(-10, 10, grid_spacing) * units.R_earth
    z_axis = np.arange(-5, 5, grid_spacing) * units.R_earth
    
    t_grid, x_grid, y_grid, z_grid = np.meshgrid(
        t_axis, y_axis, y_axis, z_axis,
        indexing='ij'
    )
    axes = libgputrace.RectilinearAxes(t_axis, x_axis, y_axis, z_axis)
    
    print('Grid Shape:', x_grid.shape)
    
    # Instantiate particle state and parallel velocity
    #pos_x = np.array([6.6]) * constants.R_earth
    pos_x = np.linspace(6, 9, 100_000) * constants.R_earth
    pos_y = np.zeros(pos_x.shape) * constants.R_earth
    pos_z = np.zeros(pos_x.shape) * constants.R_earth
    #vpar = np.ones(pos_x.shape) * 1e-2 * (constants.R_earth / units.s)
    #magnetic_moment = np.ones(pos_x.shape) * 1e-36 * 1e9 * units.A * constants.R_earth**2

    vtotal = 0.5 * constants.c
    pitch_angle = 45
    gamma = 1 / np.sqrt(1 - (vtotal/constants.c)**2)
    pperp = np.ones(pos_x.shape) * gamma * constants.m_e * np.sin(np.deg2rad(pitch_angle)) * vtotal
    ppar = np.ones(pos_x.shape) * gamma * constants.m_e * np.cos(np.deg2rad(pitch_angle)) * vtotal
    magnetic_moment = gamma * pperp**2 / (2 * constants.m_e * 100 * units.nT)

    print(f'ppar={ppar.to(constants.m_e * constants.c).value} m_e * c')
    print(f'M={magnetic_moment.to(units.MeV/units.Gauss)}')
    charge = - elementary_charge * units.C
    mass = constants.m_e
    particle_state = libgputrace.ParticleState(
        pos_x, pos_y, pos_z, 
        ppar, magnetic_moment, mass, charge
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
    B = B * units.nT
    Ex = np.zeros(Bx.shape) * units.mV/units.m
    Ey = np.zeros(Bx.shape) * units.mV/units.m
    Ez = np.zeros(Bx.shape) * units.mV/units.m

    field_model = libgputrace.RectilinearFieldModel(
        Bx, By, Bz, B, Ex, Ey, Ez, mass, charge, axes
    )

    # Call the trace routine
    start_time = time.time()
    hist = libgputrace.trace_trajectory(config, particle_state, field_model)
    end_time = time.time()
    
    print('took ', end_time - start_time, 's')
    
    # Write output for visualization
    d = {}
    
    for i in range(0, 500, 50):
    #for i in range(0, particle_state.x.size, 50):
        d[f't{i}'] = hist.t[:, i]
        d[f'x{i}'] = hist.x[:, i]
        d[f'y{i}'] = hist.y[:, i]
        d[f'z{i}'] = hist.z[:, i] 
        d[f'ppar{i}'] = hist.ppar[:, i] 
        d[f'B{i}'] = hist.B[:, i]
        d[f'W{i}'] = hist.W[:, i]
        d[f'h{i}'] = hist.h[:, i]
        
        pd.DataFrame(d).to_csv('data/test_dipole.csv')


if __name__ == '__main__':
    main()
