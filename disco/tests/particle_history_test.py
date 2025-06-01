"""Tests for the disco._particle_history module."""
import numpy as np
from astropy import units, constants
from disco._particle_history import ParticleHistory


def test_particle_history_units():
    """Test ParticleHistory handles units properly."""
    # Define base variables with astropy units
    dim_t = np.array([1.0, 2.0])
    dim_x = np.array([1.0, 2.0])
    dim_y = np.array([2.0, 3.0])
    dim_z = np.array([3.0, 4.0])
    dim_ppar = np.array([10.0, 20.0])
    dim_B = np.array([50.0, 60.0])
    dim_W = np.array([1.0, 2.0])
    dim_h = np.array([0.1, 0.2])
    mass = constants.m_e
    charge = constants.e.si

    # Create ParticleHistory instance
    hist = ParticleHistory(dim_t, dim_x, dim_y, dim_z, dim_ppar, dim_B, dim_W, dim_h, mass, charge)

    # Check that all attributes exist
    for attr in ["t", "x", "y", "z", "ppar", "B", "W", "h"]:
        assert hasattr(hist, attr)

    # Check that attributes have units
    assert hist.t.unit.is_equivalent(units.s)
    assert hist.x.unit.is_equivalent(constants.R_earth.unit)
    assert hist.y.unit.is_equivalent(constants.R_earth.unit)
    assert hist.z.unit.is_equivalent(constants.R_earth.unit)
    assert hist.ppar.unit.is_equivalent(units.keV * units.s / units.m)
    assert hist.B.unit.is_equivalent(units.nT)
    assert hist.W.unit.is_equivalent(units.keV)
    assert hist.h.unit.is_equivalent(units.s)

    # Check shape
    assert hist.t.shape == dim_t.shape
    assert hist.x.shape == dim_x.shape
    assert hist.y.shape == dim_y.shape
    assert hist.z.shape == dim_z.shape
    assert hist.B.shape == dim_B.shape

    # Check that W is undimensionalized to keV
    assert hist.W.unit.is_equivalent(units.keV)
