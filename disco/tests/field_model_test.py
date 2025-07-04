"""Tests for the disco._field_model module."""
import tempfile

from astropy import units, constants
import h5py
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


def create_hdf5_with_shape_error(axis_error=False, field_error=False):
    """Create a temp HDF5 file, optionally with a shape error in axis or field arrays."""
    # Create a temporary HDF5 file with the required structure
    nx, ny, nz, nt = 2, 2, 2, 2

    hdf_path = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    hdf = h5py.File(hdf_path.name, "w")

    # Correct shapes
    hdf["Bx"] = np.ones((nx, ny, nz, nt))
    hdf["By"] = np.ones((nx, ny, nz, nt)) * 2
    hdf["Bz"] = np.ones((nx, ny, nz, nt)) * 3
    hdf["Ex"] = np.ones((nx, ny, nz, nt)) * 4
    hdf["Ey"] = np.ones((nx, ny, nz, nt)) * 5
    hdf["Ez"] = np.ones((nx, ny, nz, nt)) * 6
    hdf["xaxis"] = np.linspace(1, 2, nx)
    hdf["yaxis"] = np.linspace(1, 2, ny)
    hdf["zaxis"] = np.linspace(1, 2, nz)
    hdf["taxis"] = np.linspace(0, 1, nt)
    hdf["r_inner"] = 1.0

    if axis_error:
        # Make xaxis 2D instead of 1D
        del hdf["xaxis"]
        hdf["xaxis"] = np.ones((nx, 2))

    if field_error:
        # Make Bx shape wrong
        del hdf["Bx"]
        hdf["Bx"] = np.ones((nx, ny, nz, 1))

    hdf.close()

    return hdf_path


def create_test_hdf5_file(nx=2, ny=2, nz=2, nt=2):
    """Create a temporary HDF5 file with test data"""
    hdf_path = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)

    hdf = h5py.File(hdf_path.name, "w")
    hdf["Bx"] = np.ones((nx, ny, nz, nz))
    hdf["By"] = np.ones((nx, ny, nz, nz)) * 2
    hdf["Bz"] = np.ones((nx, ny, nz, nz)) * 3
    hdf["Ex"] = np.ones((nx, ny, nz, nz)) * 4
    hdf["Ey"] = np.ones((nx, ny, nz, nz)) * 5
    hdf["Ez"] = np.ones((nx, ny, nz, nz)) * 6
    hdf["xaxis"] = np.linspace(1, 2, nx)
    hdf["yaxis"] = np.linspace(1, 2, ny)
    hdf["zaxis"] = np.linspace(1, 2, nz)
    hdf["taxis"] = np.linspace(0, 1, nz)
    hdf["r_inner"] = 1.0
    hdf.close()

    return hdf_path


def test_generic_hdf5_field_model_loads():
    """Test that GenericHdf5FieldModel loads data from HDF5 and sets units correctly."""
    hdf_path = create_test_hdf5_file()
    model = FieldModel.load(hdf_path.name)

    # Check shapes
    assert model.Bx.shape == (2, 2, 2, 2)
    assert model.By.shape == (2, 2, 2, 2)
    assert model.Bz.shape == (2, 2, 2, 2)
    assert model.Ex.shape == (2, 2, 2, 2)
    assert model.Ey.shape == (2, 2, 2, 2)
    assert model.Ez.shape == (2, 2, 2, 2)

    # Check axes
    assert model.axes.x.shape[0] == 2
    assert model.axes.y.shape[0] == 2
    assert model.axes.z.shape[0] == 2
    assert model.axes.t.shape[0] == 2

    # Check r_inner
    assert model.axes.r_inner == 1.0 * units.R_earth

    hdf_path.close()


def test_generic_hdf5_field_model_units():
    """Test that FieldModel.load applies correct astropy units."""
    # Create a temporary HDF5 file and load it
    hdf_path = create_test_hdf5_file()
    model = FieldModel.load(hdf_path.name)

    # Check units
    assert hasattr(model.Bx, "unit") and model.Bx.unit.is_equivalent(units.nT)
    assert hasattr(model.Ex, "unit") and model.Ex.unit.is_equivalent(units.mV / units.m)
    assert hasattr(model.axes.x, "unit") and model.axes.x.unit.is_equivalent(units.R_earth)
    assert hasattr(model.axes.t, "unit") and model.axes.t.unit.is_equivalent(units.s)

    hdf_path.close()


def test_invalid_axis_shape_raises():
    """Test that ValueError is raised for bad axis shape."""
    hdf_path = create_hdf5_with_shape_error(axis_error=True)

    with pytest.raises(ValueError):
        FieldModel.load(hdf_path.name)

    hdf_path.close()


def test_invalid_field_shape_raises():
    """Test that ValueError is raised for bad field shape."""
    hdf_path = create_hdf5_with_shape_error(field_error=True)

    with pytest.raises(ValueError):
        FieldModel.load(hdf_path.name)

    hdf_path.close()
