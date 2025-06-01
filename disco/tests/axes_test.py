import numpy as np
import cupy as cp
from astropy import units
from disco._axes import Axes, DimensionalizedAxes, Neighbors
from disco._dimensionalization import dim_time


def test_axes_init_and_attributes():
    """Test Axes initialization and attribute assignment."""
    # Specify base variables with astropy units
    x = np.linspace(1, 2, 3) * units.R_earth
    y = np.linspace(2, 3, 4) * units.R_earth
    z = np.linspace(3, 4, 5) * units.R_earth
    t = np.linspace(0, 1, 6) * units.s
    r_inner = 1.0 * units.R_earth

    # Create instance
    axes = Axes(x, y, z, t, r_inner)

    # Check values are stored unmodified
    assert np.all(axes.x == x)
    assert np.all(axes.y == y)
    assert np.all(axes.z == z)
    assert np.all(axes.t == t)
    assert axes.r_inner == r_inner


def test_dimensionalized_axes_gpu_arrays():
    """Test DimensionalizedAxes stores arrays on GPU and dimensionalizes
    correctly.
    """
    # Specify base variables with astropy units
    x = np.linspace(1, 2, 3) * units.R_earth
    y = np.linspace(2, 3, 4) * units.R_earth
    z = np.linspace(3, 4, 5) * units.R_earth
    t = np.linspace(0, 1, 6) * units.s
    r_inner = 1.0 * units.R_earth

    # Create Axes and DimensionalizedAxes instances
    axes = Axes(x, y, z, t, r_inner)
    dimmed_axes = DimensionalizedAxes(axes)

    # Check types are cupy arrays
    assert isinstance(dimmed_axes.x, cp.ndarray)
    assert isinstance(dimmed_axes.y, cp.ndarray)
    assert isinstance(dimmed_axes.z, cp.ndarray)
    assert isinstance(dimmed_axes.t, cp.ndarray)

    # Check values are dimensionless (no units)
    assert not hasattr(dimmed_axes.x, "unit")
    assert not hasattr(dimmed_axes.y, "unit")
    assert not hasattr(dimmed_axes.z, "unit")
    assert not hasattr(dimmed_axes.t, "unit")

    # Check values are correct
    assert np.allclose(dimmed_axes.x, x.value)
    assert np.allclose(dimmed_axes.y, y.value)
    assert np.allclose(dimmed_axes.z, z.value)
    assert np.allclose(dimmed_axes.t, dim_time(t))
    assert np.allclose(dimmed_axes.r_inner, r_inner.value)


def test_get_neighbors_shape_and_type():
    """Test get_neighbors returns _Neighbors with correct shapes and types."""
    # Specify base variables with astropy units
    x = np.linspace(1, 2, 3) * units.R_earth
    y = np.linspace(2, 3, 4) * units.R_earth
    z = np.linspace(3, 4, 5) * units.R_earth
    t = np.linspace(0, 1, 6) * units.s
    r_inner = 1.0 * units.R_earth

    # Create Axes and DimensionalizedAxes instances
    axes = Axes(x, y, z, t, r_inner)
    dim_axes = DimensionalizedAxes(axes)

    # Query neighbors
    t_query = cp.array([0.1, 0.5])
    y_query = cp.array([[1.1, 2.1, 3.1], [1.5, 2.5, 3.5]])
    neighbors = dim_axes.get_neighbors(t_query, y_query)

    # Check that return value is correct class
    assert isinstance(neighbors, Neighbors)

    # Check neighbors fields are CuPy arrays
    assert isinstance(neighbors.field_i, cp.ndarray)
    assert isinstance(neighbors.field_j, cp.ndarray)
    assert isinstance(neighbors.field_k, cp.ndarray)
    assert isinstance(neighbors.field_l, cp.ndarray)

    # Check they found the correct indices
    assert np.allclose(neighbors.field_i.get(), np.array([1, 1]))
    assert np.allclose(neighbors.field_j.get(), np.array([1, 2]))
    assert np.allclose(neighbors.field_k.get(), np.array([1, 2]))
    assert np.allclose(neighbors.field_l.get(), np.array([1, 1]))
