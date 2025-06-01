"Tests for the disco.readers module" ""
import tempfile

from astropy import units
import h5py
import numpy as np
import pytest

from disco.readers import GenericHdf5FieldModel, InvalidReaderShapeError


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
        del hdf['xaxis']
        hdf["xaxis"] = np.ones((nx, 2))

    if field_error:
        # Make Bx shape wrong
        del hdf['Bx']
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
