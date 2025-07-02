"""Module for just the FieldModelLoader and subclasses."""

import cupy as cp


class FieldModelLoader:
    """Abstract base class for field model loaders.

    Has the ability to pause interpolation if the required time slice
    of field data is not loaded.
    """

    def multi_interp(self, t, y, stopped_cutoff):
        """Interpolate field values at given positions.

        Warning
        -------
        The CuPy arrays in the output arrays are atttached
        to this instance. Subsequent calls will overwrite
        them. If you want to avoid this, call .copy() on
        each array.

        Paramaters
        ----------
        t: cupy array
          Vector of dimensionalized particle times
        y: cupy array
           Vector of shape (npart, 5) of particle states
        stopped_cutoff: int
           Cutoff index for partiles that no longer require processing

        Returns
        -------
        intep_result: _MultiInterpResult
        paused: cupy array (boolean)
        """
        raise NotImplementedError()


class StaticFieldModelLoader(FieldModelLoader):
    """Wraps a DimensionalizedFieldModel to provide a FieldModelLoader
    interface.

    Never pauses particles, because all time slices are always available.
    """

    def __init__(self, field_model):
        """Get an instance that is dimensionalized and stored on the GPU.

        Warning
        --------
        This class is not threadsafe.

        Parameters
        ----------
        field_model: instance of DimensionalizedFieldModel
        """
        self.field_model = field_model
        self.axes = field_model.axes

    def multi_interp(self, t, y, stopped_cutoff):
        """Interpolate field values at given positions.

        Warning
        -------
        The CuPy arrays in the output arrays are atttached
        to this instance. Subsequent calls will overwrite
        them. If you want to avoid this, call .copy() on
        each array.

        Paramaters
        ----------
        t: cupy array
          Vector of dimensionalized particle times
        y: cupy array
           Vector of shape (npart, nstate) of particle states
        stopped_cutoff: int
           Cutoff index for partiles that no longer require processing

        Returns
        -------
        intep_result: _MultiInterpResult
        paused: cupy array (boolean)
        """
        paused = cp.zeros(y.shape[0], dtype=bool)
        interp_result = self.field_model.multi_interp(t, y, paused, stopped_cutoff)

        return interp_result, paused
