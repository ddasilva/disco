"Tests for the disco.readers module" ""
from datetime import datetime
import os
import tempfile

from astropy import units
import h5py
import numpy as np
import pytest

from disco import FieldModel
from disco.readers import GenericHdf5FieldModel, InvalidReaderShapeError, SwmfCdfFieldModelDataset
from disco.tests import testing_utils


def create_swmf_directory():
    """Create a temporary directory for SWMF test files."""
    temp_dir = tempfile.TemporaryDirectory()

    # List of dummy SWMF CDF files to create
    swmf_files = [
        "3d__var_1_e20151202-050000-000.out.cdf",
        "3d__var_1_e20151202-050100-002.out.cdf",
        "3d__var_1_e20151202-050200-014.out.cdf",
        "3d__var_1_e20151202-050300-035.out.cdf",
        "3d__var_1_e20151202-050400-029.out.cdf",
        "3d__var_1_e20151202-050500-032.out.cdf",
        "3d__var_1_e20151202-050600-005.out.cdf",
        "3d__var_1_e20151202-050700-024.out.cdf",
        "3d__var_1_e20151202-050800-013.out.cdf",
        "3d__var_1_e20151202-050900-009.out.cdf",
    ]

    # Write files with dummy content
    for swmf_file in swmf_files:
        swmf_file_path = os.path.join(temp_dir.name, swmf_file)

        with open(swmf_file_path, "w") as fh:
            fh.write("This is a dummy SWMF CDF file for testing purposes.")

    return temp_dir, swmf_files


def test_swmf_cdf_dataset_length():
    """Test that the length of the SWMF CDF dataset matches the number of files."""
    temp_dir, swmf_files = create_swmf_directory()
    dataset = SwmfCdfFieldModelDataset(f"{temp_dir.name}/*.cdf")

    assert len(dataset) == len(swmf_files), "Dataset length should match number of CDF files"


def test_swmf_cdf_dataset_time_axis():
    """Test that the time axis is correctly populated from the filenames."""
    temp_dir, swmf_files = create_swmf_directory()
    dataset = SwmfCdfFieldModelDataset(f"{temp_dir.name}/*.cdf")

    got_times = dataset.get_time_axis()

    # Check that the time axis has the correct length
    assert len(got_times) == len(swmf_files), "Time axis length should match dataset length"

    # Check that the time axis is in seconds
    assert got_times.unit.is_equivalent(units.s), "Time axis should have units of seconds"

    # Check that the time axis is what is expected
    expected_times = np.array([i for i in range(len(swmf_files))]) * units.minute
    assert np.all(got_times == expected_times), "Time axis values should match expected values"

    # Check that the first timestamp is zero
    assert dataset.get_time_axis()[0] == 0 * units.s, "First timestamp should be zero"


@pytest.mark.slow
def test_swmf_cdf_getitem():
    """Tests loading a SWMF CDF file using the Dataset's __getitem__ method."""
    swmf_cdf_file = testing_utils.get_swmf_cdf_file()

    # Use a very downsampled grid for testing to make tests go faster
    dataset = SwmfCdfFieldModelDataset(swmf_cdf_file, grid_downsample=8, verbose=0)

    # Get first item
    field_model = dataset[0]

    assert isinstance(field_model, FieldModel), "Should return a FieldModel instance"
    assert field_model.axes.t.size == 1
    assert field_model.axes.t == 0 * units.s, "Time axis should be zero for first item"
    assert field_model.Bx.shape == (64, 19, 19, 1)

    # Get standard deviations of B field components for regridding regressions
    B_std_got = [field_model.Bx.std(), field_model.By.std(), field_model.Bz.std()]
    B_std_expected = [1.29673669 * units.nT, 0.71844024 * units.nT, 0.87466463 * units.nT]
    B_std_threshold = 1e-4 * units.nT

    for got, expected in zip(B_std_got, B_std_expected):
        assert np.abs(got - expected) < B_std_threshold, f"Field model B field std should match expected value: {got} != {expected}"

    # Get standard deviations of E field components for regridding regressions
    E_std_got = [field_model.Ex.std(), field_model.Ey.std(), field_model.Ez.std()]
    E_std_expected = [
        0.00412988 * units.mV/units.m,
        0.0034817 * units.mV/units.m,
        0.01345428 * units.mV/units.m,
    ] 
    E_std_threshold = 1e-4 * units.mV/units.m

    for got, expected in zip(E_std_got, E_std_expected):
        assert np.abs(got - expected) < E_std_threshold, f"Field model E field std should match expected value: {got} != {expected}"


def test_swmf_cdf_dataset_invalid_glob():
    """Test that an error is raised if the glob pattern does not match any CDF files."""
    # Create a dataset with a glob pattern that matches no files
    with pytest.raises(FileNotFoundError):
        SwmfCdfFieldModelDataset("/nonexistance/*.cdf")


def test_swmf_cdf_dataset_t0():
    """Test that the t0 parameter is correctly applied to the time axis."""
    temp_dir, swmf_files = create_swmf_directory()
    t0 = datetime(2015, 12, 2, 5, 5, 0)
    dataset = SwmfCdfFieldModelDataset(f"{temp_dir.name}/*.cdf", t0=t0)

    got_times = dataset.get_time_axis()
    expected_times = np.array([i for i in range(len(swmf_files))]) * units.minute
    expected_times -= 5 * units.minute

    assert np.all(got_times == expected_times), "Time axis values should match expected values"


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
    model = GenericHdf5FieldModel(hdf_path.name)

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
    """Test that GenericHdf5FieldModel applies correct astropy units."""
    # Create a temporary HDF5 file and load it
    hdf_path = create_test_hdf5_file()
    model = GenericHdf5FieldModel(hdf_path.name)

    # Check units
    assert hasattr(model.Bx, "unit") and model.Bx.unit.is_equivalent(units.nT)
    assert hasattr(model.Ex, "unit") and model.Ex.unit.is_equivalent(units.mV / units.m)
    assert hasattr(model.axes.x, "unit") and model.axes.x.unit.is_equivalent(units.R_earth)
    assert hasattr(model.axes.t, "unit") and model.axes.t.unit.is_equivalent(units.s)

    hdf_path.close()


def test_invalid_axis_shape_raises():
    """Test that InvalidReaderShapeError is raised for bad axis shape."""
    hdf_path = create_hdf5_with_shape_error(axis_error=True)

    with pytest.raises(InvalidReaderShapeError):
        GenericHdf5FieldModel(hdf_path.name)

    hdf_path.close()


def test_invalid_field_shape_raises():
    """Test that InvalidReaderShapeError is raised for bad field shape."""
    hdf_path = create_hdf5_with_shape_error(field_error=True)

    with pytest.raises(InvalidReaderShapeError):
        GenericHdf5FieldModel(hdf_path.name)

    hdf_path.close()
