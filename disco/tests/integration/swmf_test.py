import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))  # Adjust path to disco module

from astropy import units, constants
import numpy as np

from disco import TraceConfig, trace_trajectory
from disco.readers import SwmfOutFieldModelDataset
from .integration_test_utils import setup_particle_state
from ..testing_utils import get_swmf_out_file


def test_bouncing_basic():
    """Tests bouncing a particle in the outer radiation belt."""
    config = TraceConfig(
        t_final=1 * units.s,
        h_initial=5 * units.ms,
        rtol=1e-2,
        output_freq=None,
    )
    field_model_dataset = SwmfOutFieldModelDataset(get_swmf_out_file())
    field_model = field_model_dataset[0].duplicate_in_time()

    particle_state = setup_particle_state(pos_x=6.6)

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

    assert not np.any(hist.stopped[0, :])
    assert np.all(hist.stopped[-1, :])
    assert np.all(np.abs(t - 0.99429552) < threshold)
    assert np.all(np.abs(x - 6.43677945) < threshold)
    assert np.all(np.abs(y - 0.18660342) < threshold)
    assert np.all(np.abs(z - -1.20647112) < threshold)
    assert np.all(np.abs(ppar - 0.36841251) < threshold)
    assert np.all(np.abs(B - 246.5511625) < threshold)
    assert np.all(np.abs(W - 0.26894275) < threshold)
    assert np.all(np.abs(h - 0.00938535) < threshold)