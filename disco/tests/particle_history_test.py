"""Tests for the disco._particle_history module."""
import numpy as np
from astropy import units, constants
from disco._particle_history import ParticleHistory


def test_particle_history_units():
    """Test ParticleHistory handles units properly."""
    # Define base variables
    mass = constants.m_e
    charge = constants.e.si

    t = np.array([1.0, 2.0]) * units.s
    x = np.array([1.0, 2.0]) * constants.R_earth
    y = np.array([2.0, 3.0]) * constants.R_earth
    z = np.array([3.0, 4.0]) * constants.R_earth
    ppar = np.array([10.0, 20.0]) * (units.keV * units.s / units.m)
    B = np.array([50.0, 60.0]) * units.nT
    M = np.array([1.0, 1.0]) * (units.MeV / units.nT)  # Magnetic moment
    W = np.array([1.0, 2.0]) * (mass * constants.c**2)  # Energy
    h = np.array([0.1, 0.2]) * units.s  # Time step
    stopped = np.array([False, True], dtype=bool)

    # Create ParticleHistory instance
    hist = ParticleHistory(t, x, y, z, ppar, M, B, W, h, stopped, mass, charge)

    # Check that all attributes exist
    for attr in ["t", "x", "y", "z", "ppar", "B", "W", "h", "stopped"]:
        assert hasattr(hist, attr)

    # Check shape
    assert hist.t.shape == t.shape
    assert hist.x.shape == x.shape
    assert hist.y.shape == y.shape
    assert hist.z.shape == z.shape
    assert hist.B.shape == B.shape
    assert hist.stopped.shape == stopped.shape


def test_particle_history_save_load_roundtrip():
    """Test that ParticleHistory.save() and ParticleHistory.load() are consistent."""
    import tempfile
    import numpy as np
    from astropy import units, constants
    from disco._particle_history import ParticleHistory

    # Define base variables
    t = np.array([1.0, 2.0]) * units.s
    x = np.array([1.0, 2.0]) * constants.R_earth
    y = np.array([2.0, 3.0]) * constants.R_earth
    z = np.array([3.0, 4.0]) * constants.R_earth
    ppar = np.array([10.0, 20.0]) * (units.keV * units.s / units.m)
    B = np.array([50.0, 60.0]) * units.nT
    M = np.array([1.0, 1.0]) * (units.MeV / units.nT)
    W = np.array([1.0, 2.0]) * (constants.m_e * constants.c**2)
    h = np.array([0.1, 0.2]) * units.s
    stopped = np.array([False, True], dtype=bool)
    mass = constants.m_e
    charge = constants.e.si

    # Create ParticleHistory instance
    hist = ParticleHistory(
        t=t, x=x, y=y, z=z, ppar=ppar, M=M, B=B, W=W, h=h, stopped=stopped, mass=mass, charge=charge
    )

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".h5") as temp_file:
        hist.save(temp_file.name)

        # Load from the same file
        loaded_hist = ParticleHistory.load(temp_file.name)

    # Assert that all attributes are equal
    for attr in ["t", "x", "y", "z", "ppar", "M", "B", "W", "h"]:
        expected = getattr(hist, attr).value
        got = getattr(loaded_hist, attr).value
        np.testing.assert_allclose(expected, got, err_msg=attr)

        expected_units = getattr(hist, attr).unit
        got_units = getattr(loaded_hist, attr).unit
        assert (
            expected_units == got_units
        ), f"Units mismatch for {attr}: {expected_units} != {got_units}"

    assert np.all(hist.stopped == loaded_hist.stopped)

    # Assert mass and charge are equal
    assert hist.mass == loaded_hist.mass
    assert hist.charge == loaded_hist.charge
