

import libgputrace
from pyhdf.SD import SD, SDC
import numpy as np
from astropy import units, constants
from scipy.constants import elementary_charge
import pandas as pd
import time


def main():
    ntimes = 6
    
    # Read the B and E Fields
    hdf = SD("data/LM60-Quad_mhd_2013-03-17T00-00-00Z-field.24h.hdf", SDC.READ)    
    t_axis = hdf.select("t")[:ntimes] * units.s
    Bx = hdf.select("bx")[:ntimes] * units.nT
    By = hdf.select("by")[:ntimes] * units.nT
    Bz = hdf.select("bz")[:ntimes] * units.nT
    Ex = hdf.select("ex")[:ntimes] * units.V/units.km
    Ey = hdf.select("ey")[:ntimes] * units.V/units.km
    Ez = hdf.select("ez")[:ntimes] * units.V/units.km

    hdf.end()

    # Read the Grid
    hdf = SD("data/LM60-Quad_mhd_2013-03-grid.24h.hdf", SDC.READ)
    x_axis = hdf.select("x")[:] * units.R_earth
    y_axis = hdf.select("y")[:] * units.R_earth
    z_axis = hdf.select("z")[:] * units.R_earth

    # # Change order of axes
    axis_from = [0, 1, 2, 3]
    axis_to = [3, 2, 1, 0]

    Bx = np.moveaxis(Bx, axis_from, axis_to)
    By = np.moveaxis(By, axis_from, axis_to)
    Bz = np.moveaxis(Bz, axis_from, axis_to)
    Ex = np.moveaxis(Ex, axis_from, axis_to)
    Ey = np.moveaxis(Ey, axis_from, axis_to)
    Ez = np.moveaxis(Ez, axis_from, axis_to)

    # Print or use the data
    print("Bx:", Bx.shape)
    print("By:", By.shape)
    print("Bz:", Bz.shape)
    print("Ex:", Ex.shape)
    print("Ey:", Ey.shape)
    print("Ez:", Ez.shape)
    print("t:", t_axis.shape)
    print("x:", x_axis.shape)
    print("y:", y_axis.shape)
    print("z:", z_axis.shape)

    print("time=", t_axis)
    
    # Setup particles
    #t_final = t_axis[-2].valeu
    t_final = 1 * units.s
    config = libgputrace.TraceConfig(
        t_final=t_final,
        rtol=5e-3,
        output_freq=5,
    )
    
    #pos_x = np.linspace(6, 9, 100_000) * constants.R_earth
    pos_x = np.array([6.6]* 100_000) * constants.R_earth
    pos_y = np.zeros(pos_x.shape) * constants.R_earth
    pos_z = np.zeros(pos_x.shape) * constants.R_earth

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

    # Setup axes and field model
    axes = libgputrace.RectilinearAxes(t_axis, x_axis, y_axis, z_axis)    

    field_model = libgputrace.RectilinearFieldModel(
        Bx, By, Bz, Ex, Ey, Ez, mass, charge, axes
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
        
        pd.DataFrame(d).to_csv('data/test_lfm.csv')
    
    
if __name__ == '__main__':
    main()
