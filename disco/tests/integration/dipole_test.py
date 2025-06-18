import numpy as np

from astropy import constants, units
from matplotlib import pyplot as plt
import pytest
from scipy.constants import elementary_charge

from disco import TraceConfig, Axes, FieldModel, ParticleState, trace_trajectory


def _setup_field_model(charge=-1, backwards_time=False):
    # Setup axes and grid
    grid_spacing = 0.5

    x_axis = np.arange(-10, 10, grid_spacing) * constants.R_earth
    y_axis = np.arange(-10, 10, grid_spacing) * constants.R_earth
    z_axis = np.arange(-5, 5, grid_spacing) * constants.R_earth

    if backwards_time:
        t_axis = np.arange(-5, 2) * units.s
    else:
        t_axis = np.arange(-1, 5) * units.s

    x_grid, y_grid, z_grid, t_grid = np.meshgrid(x_axis, y_axis, z_axis, t_axis, indexing="ij")
    r_inner = 1 * constants.R_earth

    axes = Axes(x_axis, y_axis, z_axis, t_axis, r_inner)

    # Setup field model (no external field)
    Bx = np.zeros(x_grid.shape) * units.nT
    By = np.zeros(Bx.shape) * units.nT
    Bz = np.zeros(Bx.shape) * units.nT
    Ex = np.zeros(Bx.shape) * units.mV / units.m
    Ey = np.zeros(Bx.shape) * units.mV / units.m
    Ez = np.zeros(Bx.shape) * units.mV / units.m

    field_model = FieldModel(Bx, By, Bz, Ex, Ey, Ez, axes)

    return field_model


def _setup_particle_state(
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


def test_bouncing_basic():
    """Tests bouncing a particle in the outer radiation belt."""
    config = TraceConfig(
        t_final=1 * units.s,
        h_initial=5 * units.ms,
        rtol=1e-2,
        output_freq=None,
    )
    field_model = _setup_field_model()
    particle_state = _setup_particle_state(pos_x=6.6)

    hist = trace_trajectory(config, particle_state, field_model, verbose=0)

    threshold = 1e-2

    t = hist.t[-1, :].to(units.s).value
    x = hist.x[-1, :].to(constants.R_earth).value
    y = hist.y[-1, :].to(constants.R_earth).value
    z = hist.z[-1, :].to(constants.R_earth).value
    ppar = hist.ppar[-1, :].to(constants.m_e * constants.c).value
    B = hist.B[-1, :].to(units.nT).value
    W = hist.W[-1, :].to(constants.m_e * constants.c**2).value
    h = hist.h[-1, :].to(units.s).value

    assert np.all(np.abs(t - 0.99920958) < threshold)
    assert np.all(np.abs(x - 6.34677013) < threshold)
    assert np.all(np.abs(y - 0.36272624) < threshold)
    assert np.all(np.abs(z - 1.01137163) < threshold)
    assert np.all(np.abs(ppar - 0.37732395) < threshold)
    assert np.all(np.abs(B - 120.45078467) < threshold)
    assert np.all(np.abs(W - 0.17225467) < threshold)
    assert np.all(np.abs(h - 0.00718319) < threshold)


def setup_plotting(output_freq):
    """Sets up a bouncing particle in the outer radiation belt for
    testing plotting
    """
    config = TraceConfig(
        t_final=1 * units.s,
        h_initial=5 * units.ms,
        rtol=1e-2,
        output_freq=output_freq,
    )
    field_model = _setup_field_model()
    particle_state = _setup_particle_state(pos_x=6.6)

    hist = trace_trajectory(config, particle_state, field_model, verbose=0)

    return hist


@pytest.mark.parametrize("method_name", ["plot_xy", "plot_xz", "plot_yz"])
def test_bouncing_plotting_xy(method_name):
    """Tests bouncing a particle in the outer radiation belt and then
    plotting the results.

    Checks that no exceptions are thrown during plotting
    """
    hist = setup_plotting(output_freq=1)

    method = getattr(hist, method_name)

    ax = method()
    assert ax is not None

    ax = method(sample=1)
    ax = method(inds=0)
    ax = method(inds=[0, 1])
    ax = method(endpoints=True)
    ax = method(earth=False)

    with pytest.raises(IndexError, match="out of bounds for the number of particles"):
        ax = method(inds=1000)

    with pytest.raises(IndexError, match="out of bounds for the number of particles"):
        ax = method(inds=np.arange(1000))

    # Need to close figures, otherwise matplotlib will complain
    plt.close("all")


def test_bouncing_history():
    """Tests bouncing a particle in the outer radiation belt
    and plotting trajectory history.
    """
    config = TraceConfig(
        t_final=1 * units.s,
        h_initial=5 * units.ms,
        rtol=1e-2,
        output_freq=1,
    )
    field_model = _setup_field_model()
    particle_state = _setup_particle_state(pos_x=6.6)

    hist = trace_trajectory(config, particle_state, field_model, verbose=0)

    min_shape = 100
    assert hist.t.shape[0] > min_shape
    assert hist.x.shape[0] > min_shape
    assert hist.y.shape[0] > min_shape
    assert hist.z.shape[0] > min_shape
    assert hist.ppar.shape[0] > min_shape
    assert hist.B.shape[0] > min_shape
    assert hist.W.shape[0] > min_shape
    assert hist.h.shape[0] > min_shape


def test_bouncing_stop_cond():
    """Tests bouncing a particle in the outer radiation belt
    with a stopping condition to stop when z > 1
    """
    zmax = 1

    def stop_cond(y, t, field_model):
        return y[:, 2] > zmax

    config = TraceConfig(
        t_final=1 * units.s,
        h_initial=5 * units.ms,
        rtol=1e-2,
        output_freq=None,
        stopping_conditions=[stop_cond],
    )
    field_model = _setup_field_model()
    particle_state = _setup_particle_state(pos_x=6.6)

    hist = trace_trajectory(config, particle_state, field_model, verbose=0)

    # Integration will stop AFTER stop_cond is true
    test_threshold = 1.1
    z = hist.z[:, :].to(constants.R_earth).value

    assert np.all(z < test_threshold)


def test_oob_zmax():
    """Tests bouncing a particle in the outer radiation belt
    with particle that travels out of bounds (above zmax)
    """
    config = TraceConfig(
        t_final=1 * units.s,
        h_initial=5 * units.ms,
        rtol=1e-2,
        output_freq=None,
    )
    field_model = _setup_field_model()
    particle_state = _setup_particle_state(pos_x=6.6)

    hist = trace_trajectory(config, particle_state, field_model, verbose=0)

    z_got = hist.z[:, :-1].to(constants.R_earth).value
    z_expected = field_model.axes.z[-1].to(constants.R_earth).value

    assert np.all(z_got < z_expected)


def test_oob_rinner():
    """Tests bouncing a particle in the outer radiation belt
    with particle that travels out of bounds (below rinner)
    """
    config = TraceConfig(
        t_final=1 * units.s,
        h_initial=5 * units.ms,
        rtol=1e-2,
        output_freq=None,
    )
    field_model = _setup_field_model()
    particle_state = _setup_particle_state(pos_x=6.6, pitch_angle=0)

    hist = trace_trajectory(config, particle_state, field_model, verbose=0)
    r = np.sqrt(hist.x**2 + hist.y**2 + hist.z**2).to(constants.R_earth)

    assert np.all(r > field_model.axes.r_inner)


def test_reorder_does_not_affect_history():
    """Starts particles on dipole between x=-4 and x=-8 through
    part of a bounce with reordering every step. Checks at the
    end that they are still sorted by x coordinate (they should
    stay on the same field line).
    """
    config = TraceConfig(
        t_final=250 * units.ms,
        h_initial=5 * units.ms,
        rtol=1e-2,
        output_freq=1,
        reorder_freq=1,
    )
    field_model = _setup_field_model(backwards_time=True)

    # Setup custom particle state
    pos_x = -np.arange(4, 8, 0.1) * constants.R_earth
    npart = pos_x.shape
    pos_y = np.zeros(npart) * constants.R_earth
    pos_z = np.zeros(npart) * constants.R_earth
    vtotal = 0.1 * constants.c
    gamma = 1 / np.sqrt(1 - (vtotal / constants.c) ** 2)
    m = constants.m_e
    ppar = np.ones(npart) * gamma * m * vtotal
    magnetic_moment = 0 * units.MeV / units.nT
    charge = -elementary_charge * units.C
    particle_state = ParticleState(pos_x, pos_y, pos_z, ppar, magnetic_moment, m, charge)

    # Trace trajectory
    hist = trace_trajectory(config, particle_state, field_model, verbose=0)

    assert np.all(np.diff(hist.x) < 0)


def test_bouncing_integrate_backwards():
    """Tests bouncing a particle in the outer radiation belt
    with particle integrating backwards in time
    """
    config = TraceConfig(
        t_final=-1 * units.s,
        h_initial=5 * units.ms,
        rtol=1e-2,
        output_freq=1,
        integrate_backwards=True,
    )
    field_model = _setup_field_model(backwards_time=True)
    particle_state = _setup_particle_state(pos_x=6.6, pitch_angle=0)

    hist = trace_trajectory(config, particle_state, field_model, verbose=0)

    assert hist.x.shape[0] > 100
