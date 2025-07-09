"""Module for just the FieldModelLoader and subclasses."""

import cupy as cp
import numpy as np

from disco._axes import Axes
from disco._dimensionalization import dim_time
from disco._field_model import FieldModel


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


class LazyFieldModelLoader(FieldModelLoader):
    """Wraps a FieldModelDataset to provide lazy loadin of field models.

    Tis class pauses particles if the required time slice
    of field data is not loaded. This is useful for large datasets
    where not all time slices are available at once, such as
    in the case of simulations.
    """

    def __init__(self, dataset, config, mass, charge, window_size=2, verbose=1):
        """Get an instance that is dimensionalized and stored on the GPU.

        Parameters
        ----------
        dataset : `disco.readers.FieldModelDataset`
           Dataset that reads field models from disk on demand
        config : `disco.TraceConfig`
           Configuration for the trace
        mass : scalar with units
           Mass of the particle
        charge : scalar with units
           Charge of the particle
        window_size : int
           Number of time slices to load at once
        verbose: int
           Verbosity level for logging. Set to 0 to supress output.
        """
        self.field_model_dataset = dataset
        self.time_axis = dataset.get_time_axis()
        self.time_axis_dim = dim_time(self.time_axis)
        self.config = config
        self.mass = mass
        self.charge = charge
        self.window_size = window_size
        self.verbose = verbose

        self.integration_dir = -1 if config.integrate_backwards else 1
        self.cache = {}

        # Load the starting set of field models
        if config.integrate_backwards:
            self.end_index = np.searchsorted(self.time_axis, config.t_initial, side="left")
            self.start_index = self.end_index - window_size + 1
            if self.start_index < 0:
                self.start_index = 0
        else:
            self.start_index = np.searchsorted(self.time_axis, config.t_initial, side="right") - 1
            self.end_index = self.start_index + window_size - 1
            if self.end_index >= len(self.time_axis):
                self.end_index = len(self.time_axis) - 1

        self._load_field_models()

    def _load_field_models(self):
        """Load field models for the given time slice range.

        Sets self.field_model and self.axes attributes.
        """
        # Get the indices of the field models to load
        indices = range(self.start_index, self.end_index + 1)

        # Load the field models for the given indices
        for i in indices:
            if i not in self.cache:
                if self.verbose > 0:
                    print(f"Loading field model for index {i} at time {self.time_axis[i]}")

                self.cache[i] = self.field_model_dataset[i]

        # Remove field models that are no longer in the range
        for key in list(self.cache.keys()):
            if key < indices[0] or key > indices[-1]:
                del self.cache[key]

        # Stack the field models into a single FieldModel
        Bx = np.stack([self.cache[i].Bx for i in indices], axis=3)
        By = np.stack([self.cache[i].By for i in indices], axis=3)
        Bz = np.stack([self.cache[i].Bz for i in indices], axis=3)
        Ex = np.stack([self.cache[i].Ex for i in indices], axis=3)
        Ey = np.stack([self.cache[i].Ey for i in indices], axis=3)
        Ez = np.stack([self.cache[i].Ez for i in indices], axis=3)

        sample_axes = self.cache[indices[0]].axes
        axes = Axes(
            x=sample_axes.x,
            y=sample_axes.y,
            z=sample_axes.z,
            t=self.time_axis[self.start_index : self.end_index + 1],
            r_inner=sample_axes.r_inner,
        )

        # Store dimensionalized field model and axes in self
        self.field_model = FieldModel(
            Bx,
            By,
            Bz,
            Ex,
            Ey,
            Ez,
            axes=axes,
            B0=self.field_model_dataset.B0,
        ).dimensionalize(self.mass, self.charge)
        self.axes = self.field_model.axes

    def multi_interp(self, t, y, stopped_cutoff):
        """Interpolate field values at given positions.

        Parameters
        ----------
        t: cupy array.
           Vector of dimensionalized particle times.
        y: cupy array
           Vector of shape (npart, nstate) of ongoing particle states.
        stopped_cutoff: int
           Cutoff index for particles that no longer require processing.

        Returns
        -------
        intep_result: `_MultiInterpResult`
            Contains interpolated field values for each particle at the given times.
        paused: cupy array of booleans
            Indicates whether each particle is paused (True) or not (False). Particles
            are paused if their time is outside the range of loaded field models.

        Notes
        -----
        The CuPy arrays in the output arrays are atttached
        to this instance. Subsequent calls will overwrite
        them. If you want to avoid this, call .copy() on
        each array.
        """
        # Adjust field models currently loaded based on particle positions
        if self.config.integrate_backwards:
            unneeded_slice = cp.all(t < self.axes.t[-2])
            room_to_slide = self.start_index > 0

            if unneeded_slice and room_to_slide:
                # We can drop the last time slice, since all particles are before it
                self.start_index = max(0, self.start_index - 1)
                self.end_index = max(0, self.end_index - 1)
                self._load_field_models()
        else:
            unneeded_slice = cp.all(t > self.axes.t[1])
            room_to_slide = self.end_index < len(self.time_axis) - 1

            if unneeded_slice and room_to_slide:
                # We can drop the first time slice, since all particles are after it
                self.start_index = min(len(self.time_axis) - 1, self.start_index + 1)
                self.end_index = min(len(self.time_axis) - 1, self.end_index + 1)
                self._load_field_models()

        # Determine the paused state
        if self.config.integrate_backwards:
            paused = t < self.time_axis_dim[self.start_index]
        else:
            paused = t > self.time_axis_dim[self.end_index]

        # Perform the interpolation
        interp_result = self.field_model.multi_interp(t, y, paused, stopped_cutoff)

        # Return the interpolation result and paused state
        return interp_result, paused


class StaticFieldModelLoader(FieldModelLoader):
    """Wraps a DimensionalizedFieldModel to provide a FieldModelLoader
    interface.

    Never pauses particles, because all time slices are always available.
    """

    def __init__(self, field_model):
        """Get an instance that is dimensionalized and stored on the GPU.

        Notes
        -----
        This class is not threadsafe.

        Parameters
        ----------
        field_model : `DimensionalizedFieldModel`
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
