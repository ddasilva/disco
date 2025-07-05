import numpy as np
from astropy import units, constants
from scipy.constants import elementary_charge

from disco import ParticleState


def setup_particle_state(
    vtotal=0.5, pitch_angle=45, npart=10, pos_x=0, pos_y=0, pos_z=0, charge=-1
):

    pos_x = np.array([pos_x] * npart) * constants.R_earth
    pos_y = np.array([pos_y] * npart) * constants.R_earth
    pos_z = np.array([pos_z] * npart) * constants.R_earth

    vtotal = vtotal * constants.c
    gamma = 1 / np.sqrt(1 - (vtotal / constants.c) ** 2)

    if charge < 0:
        m = constants.m_e
    else:
        m = constants.m_p

    pperp = np.ones(npart) * gamma * m * np.sin(np.deg2rad(pitch_angle)) * vtotal
    ppar = np.ones(npart) * gamma * m * np.cos(np.deg2rad(pitch_angle)) * vtotal
    magnetic_moment = gamma * pperp**2 / (2 * m * 100 * units.nT)
    charge = charge * elementary_charge * units.C

    particle_state = ParticleState(pos_x, pos_y, pos_z, ppar, magnetic_moment, m, charge)

    return particle_state
