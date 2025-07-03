"""Tests for the disco._field_model module."""

from astropy import units, constants
import numpy as np
import pytest

from disco._field_model import FieldModel
from disco._axes import Axes


def _setup_single_timestep_field_model(B_delta=0 * units.nT, E_delta=0 * units.mV / units.m):
    """Sets up a FieldModel with a single time step for testing purposes.

    Returns
    -------
    A field model with zero magnetic and electric fields,  defined over a grid with
    a single time step.
    """
    # Setup axes and grid
    grid_spacing = 0.5

    x_axis = np.arange(-10, 10, grid_spacing) * constants.R_earth
    y_axis = np.arange(-10, 10, grid_spacing) * constants.R_earth
    z_axis = np.arange(-5, 5, grid_spacing) * constants.R_earth
    t_axis = np.array([0]) * units.s

    x_grid, _, _, _ = np.meshgrid(x_axis, y_axis, z_axis, t_axis, indexing="ij")
    r_inner = 1 * constants.R_earth

    axes = Axes(x_axis, y_axis, z_axis, t_axis, r_inner)

    # Setup field model with optional default values
    Bx = np.zeros(x_grid.shape) * units.nT + B_delta
    By = np.zeros(Bx.shape) * units.nT + B_delta
    Bz = np.zeros(Bx.shape) * units.nT + B_delta
    Ex = np.zeros(Bx.shape) * units.mV / units.m + E_delta
    Ey = np.zeros(Bx.shape) * units.mV / units.m + E_delta
    Ez = np.zeros(Bx.shape) * units.mV / units.m + E_delta

    # Create the FieldModel instance
    field_model = FieldModel(Bx, By, Bz, Ex, Ey, Ez, axes)

    return field_model


def test_duplicate_in_time_duplicates_values():
    """Test the duplicate_in_time() method of FieldModel duplicates the field
    values correctly.
    """
    field_model_original = _setup_single_timestep_field_model(
        B_delta=1 * units.nT, E_delta=2 * units.mV / units.m
    )
    field_model_changed = field_model_original.duplicate_in_time()

    # Check that the changed field models have the same values in both time steps
    for attr in ["Bx", "By", "Bz", "Ex", "Ey", "Ez"]:
        original_value = getattr(field_model_original, attr)
        changed_value = getattr(field_model_changed, attr)

        message = f"Values for {attr} do not match the original data."
        assert np.array_equal(original_value[:, :, :, 0], changed_value[:, :, :, 0]), message
        assert np.array_equal(original_value[:, :, :, 0], changed_value[:, :, :, 1]), message

        message = f"Values for {attr} do not match in both timesteps."
        assert np.array_equal(changed_value[:, :, :, 0], changed_value[:, :, :, 1]), message


def test_duplicate_in_time_correct_shape():
    """Test the duplicate_in_time() method of FieldModel generates a new
    FieldModel with correct shape.
    """
    field_model_original = _setup_single_timestep_field_model()
    field_model_changed = field_model_original.duplicate_in_time()

    assert field_model_original.axes.t.size == 1
    assert field_model_changed.axes.t.size == 2

    orig_shape = field_model_original.Bx.shape

    for attr in ["Bx", "By", "Bz", "Ex", "Ey", "Ez"]:
        got_shape = getattr(field_model_changed, attr).shape
        expected_shape = (orig_shape[0], orig_shape[1], orig_shape[2], 2)
        message = f"Shape mismatch for {attr}: {got_shape} != {expected_shape}"
        assert got_shape == expected_shape, message


def test_duplicate_in_time_raises_error_for_multiple_timesteps():
    """Tests that the duplicate_in_time() method raises an error if the
    FieldModel has more than one time step.
    """
    field_model_original = _setup_single_timestep_field_model()
    field_model_changed = field_model_original.duplicate_in_time()

    with pytest.raises(ValueError):
        field_model_changed.duplicate_in_time()
