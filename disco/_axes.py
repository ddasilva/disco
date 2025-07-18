from dataclasses import dataclass
from typing import Any

import cupy as cp

from disco._dimensionalization import dim_space, dim_time


class Axes:
    """Coordinates of a Rectilinear Grid in SM Coordinates."""

    def __init__(self, x, y, z, t, r_inner):
        """Initialize instance that is dimensionalized and stored
        on the GPU.

        Parameters
        ----------
        x: array with units
            x axis coordinates
        y: array with units
            y axis coordinates
        z: array with units
            z axis coordinates
        t: numpy array with units
            time axis coordinates
        r_inner: scalar with units
            inner boundary
        """
        assert len(x.shape) == 1
        assert len(y.shape) == 1
        assert len(z.shape) == 1
        assert len(t.shape) == 1

        self.t = t
        self.x = x
        self.y = y
        self.z = z
        self.r_inner = r_inner

    def dimensionalize(self):
        """Convert to a `DimensionalizedAxes` instance.

        Returns
        -------
        instance of `DimensionalizedAxes`
        """

        return DimensionalizedAxes(self)


class DimensionalizedAxes:
    """
    1D arrays of uniform grid axes

    Stored in dimensionalized form on the GPU.
    """

    def __init__(self, axes):
        self.x = cp.array(dim_space(axes.x))
        self.y = cp.array(dim_space(axes.y))
        self.z = cp.array(dim_space(axes.z))
        self.t = cp.array(dim_time(axes.t))
        self.r_inner = dim_space(axes.r_inner)

    def get_neighbors(self, t, y):
        """Get instance of _Neighbors specifying surrounding
        cell through indeces of upper corner

        Returns instance of _Neighbors
        """
        field_i = cp.searchsorted(self.x, y[:, 0])
        field_j = cp.searchsorted(self.y, y[:, 1])
        field_k = cp.searchsorted(self.z, y[:, 2])
        field_l = cp.searchsorted(self.t, t, side="right")

        return Neighbors(
            field_i=field_i,
            field_j=field_j,
            field_k=field_k,
            field_l=field_l,
        )


@dataclass
class Neighbors:
    """Neighbors of given particles, used for interpolation"""

    field_i: Any
    field_j: Any
    field_k: Any
    field_l: Any
