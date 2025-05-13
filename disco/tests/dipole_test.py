import numpy as np

from astropy import constants, units
from scipy.constants import elementary_charge

from .. import TraceConfig, Axes, FieldModel, ParticleState, trace_trajectory


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

    if charge < 0:
        mass = constants.m_e
    else:
        mass = constants.m_p

    charge = charge * elementary_charge * units.coulomb

    field_model = FieldModel(Bx, By, Bz, Ex, Ey, Ez, mass, charge, axes)

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

    assert np.all(np.abs(hist.t[0, :] - 46.96625915) < threshold)
    assert np.all(np.abs(hist.x[0, :] - 6.34677013) < threshold)
    assert np.all(np.abs(hist.y[0, :] - 0.36272624) < threshold)
    assert np.all(np.abs(hist.z[0, :] - 1.01137163) < threshold)
    assert np.all(np.abs(hist.ppar[0, :] - 0.37732395) < threshold)
    assert np.all(np.abs(hist.B[0, :] - -9.58897718) < threshold)
    assert np.all(np.abs(hist.W[0, :] - 0.17225467) < threshold)
    assert np.all(np.abs(hist.h[0, :] - 0.34064285) < threshold)


def test_bouncing_history():
    """Tests bouncing a particle in the outer radiation belt
    and recording trajectory history.
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
    assert np.all(hist.z[:, :] < test_threshold)


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

    assert np.all(hist.z[:, :-1] < field_model.axes.z.get()[-1])


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
    r = np.sqrt(hist.x**2 + hist.y**2 + hist.z**2)

    assert np.all(r > field_model.axes.r_inner)


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
