"""Tests for the disco._field_model module."""
import tempfile

from astropy import units, constants
import h5py
import numpy as np
import pytest

from disco._field_model import FieldModel
from disco._axes import Axes


def _setup_grid():
    # Setup axes and grid
    grid_spacing = 0.5

    x_axis = np.arange(-10, 10, grid_spacing) * constants.R_earth
    y_axis = np.arange(-10, 10, grid_spacing) * constants.R_earth
    z_axis = np.arange(-5, 5, grid_spacing) * constants.R_earth
    t_axis = np.array([0]) * units.s

    x_grid, _, _, _ = np.meshgrid(x_axis, y_axis, z_axis, t_axis, indexing="ij")

    return x_axis, y_axis, z_axis, t_axis, x_grid.shape


def _setup_single_timestep_field_model(
    B_delta=0 * units.nT, E_delta=0 * units.mV / units.m, extra_fields=None
):
    """Sets up a FieldModel with a single time step for testing purposes.

    Returns
    -------
    A field model with zero magnetic and electric fields,  defined over a grid with
    a single time step.
    """
    x_axis, y_axis, z_axis, t_axis, shape = _setup_grid()
    r_inner = 1 * constants.R_earth

    axes = Axes(x_axis, y_axis, z_axis, t_axis, r_inner)

    # Setup field model with optional default values
    Bx = np.zeros(shape) * units.nT + B_delta
    By = np.zeros(Bx.shape) * units.nT + B_delta
    Bz = np.zeros(Bx.shape) * units.nT + B_delta
    Ex = np.zeros(Bx.shape) * units.mV / units.m + E_delta
    Ey = np.zeros(Bx.shape) * units.mV / units.m + E_delta
    Ez = np.zeros(Bx.shape) * units.mV / units.m + E_delta

    # Create the FieldModel instance
    field_model = FieldModel(Bx, By, Bz, Ex, Ey, Ez, axes, extra_fields=extra_fields)

    return field_model


def _do_round_trip(include_extra_fields=False):
    """Creates a FieldModel, saves it to a temporary HDF5 file, and loads it back."""
    if include_extra_fields:
        x_axis, y_axis, z_axis, t_axis, shape = _setup_grid()
        n_value = 1.5
        p_value = 8.2

        extra_fields = {
            "n": np.zeros(shape) + n_value,
            "p": np.zeros(shape) + p_value,
        }

    else:
        extra_fields = {}

    field_model = _setup_single_timestep_field_model(extra_fields=extra_fields)
    hdf_path = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    field_model.save(hdf_path.name)

    # Load the model back from the HDF5 file
    loaded_model = FieldModel.load(hdf_path.name)

    return field_model, loaded_model


def test_field_model_round_trip():
    """Test that a FieldModel can be saved and loaded correctly."""
    field_model, loaded_model = _do_round_trip()

    assert field_model.Bx.shape == loaded_model.Bx.shape
    assert field_model.By.shape == loaded_model.By.shape
    assert field_model.Bz.shape == loaded_model.Bz.shape
    assert field_model.Ex.shape == loaded_model.Ex.shape
    assert field_model.Ey.shape == loaded_model.Ey.shape
    assert field_model.Ez.shape == loaded_model.Ez.shape

    # Check axes
    assert np.array_equal(field_model.axes.x, loaded_model.axes.x)
    assert np.array_equal(field_model.axes.y, loaded_model.axes.y)
    assert np.array_equal(field_model.axes.z, loaded_model.axes.z)
    assert np.array_equal(field_model.axes.t, loaded_model.axes.t)

    # Check r_inner
    assert field_model.axes.r_inner == loaded_model.axes.r_inner


def test_field_model_round_trip_with_extra_fields():
    """Test that a FieldModel can be saved and loaded correctly, with extra fields."""
    field_model, loaded_model = _do_round_trip(include_extra_fields=True)

    assert field_model.Bx.shape == loaded_model.Bx.shape
    assert field_model.By.shape == loaded_model.By.shape
    assert field_model.Bz.shape == loaded_model.Bz.shape
    assert field_model.Ex.shape == loaded_model.Ex.shape
    assert field_model.Ey.shape == loaded_model.Ey.shape
    assert field_model.Ez.shape == loaded_model.Ez.shape

    # Check extra fields
    for key in field_model.extra_fields:
        assert key in loaded_model.extra_fields
        assert field_model.extra_fields[key].shape == loaded_model.extra_fields[key].shape

    # Check axes
    assert np.array_equal(field_model.axes.x, loaded_model.axes.x)
    assert np.array_equal(field_model.axes.y, loaded_model.axes.y)
    assert np.array_equal(field_model.axes.z, loaded_model.axes.z)
    assert np.array_equal(field_model.axes.t, loaded_model.axes.t)

    # Check r_inner
    assert field_model.axes.r_inner == loaded_model.axes.r_inner


def test_extra_fields_is_empty_dict_by_default():
    """Test that the extra_fields attribute is an empty dictionary by default."""
    field_model = _setup_single_timestep_field_model(extra_fields=None)
    assert field_model.extra_fields == {}

    field_model = _setup_single_timestep_field_model(extra_fields={})
    assert field_model.extra_fields == {}


def test_extra_fields():
    """Test that the extra_fields attribute gets populated as expected."""
    x_axis, y_axis, z_axis, t_axis, shape = _setup_grid()
    n_value = 1.5
    p_value = 8.2

    extra_fields = {
        "n": np.zeros(shape) + n_value,
        "p": np.zeros(shape) + p_value,
    }

    field_model = _setup_single_timestep_field_model(extra_fields=extra_fields)
    assert len(field_model.extra_fields) == len(extra_fields)

    assert field_model.extra_fields["n"].shape == shape
    assert field_model.extra_fields["p"].shape == shape

    assert np.all(field_model.extra_fields["n"] == n_value)
    assert np.all(field_model.extra_fields["p"] == p_value)


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


def create_test_hdf5_file(nx=2, ny=2, nz=2, nt=2, extra_fields=False):
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

    if extra_fields:
        extra_fields_group = hdf.create_group("extra_fields")
        extra_fields_group["n"] = np.ones((nx, ny, nz, nz)) * 1.5
        extra_fields_group["p"] = np.ones((nx, ny, nz, nz)) * 8.2

    hdf.close()

    return hdf_path


def test_field_model_loads():
    """Test that FieldModel loads data from HDF5 and sets units correctly."""
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

    # Check no extra fields
    assert len(model.extra_fields) == 0

    hdf_path.close()


def test_field_model_extra_fields():
    """Test that FieldModel loads data from HDF5 and sets units correctly."""
    hdf_path = create_test_hdf5_file(extra_fields=True)
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

    # Check extra fields
    assert len(model.extra_fields) == 2
    assert model.extra_fields["n"].shape == (2, 2, 2, 2)
    assert model.extra_fields["p"].shape == (2, 2, 2, 2)
    assert np.all(model.extra_fields["n"] == 1.5)
    assert np.all(model.extra_fields["p"] == 8.2)

    hdf_path.close()


def test_field_model_loads_units():
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
