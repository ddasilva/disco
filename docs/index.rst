
DISCO
=======


DISCO is a tool for magnetospheric particle trajectory modeling, also known as particle tracing, on the GPU. It calculates the trajectories of particles using guiding center theory with full support for relativistic effects. The tool operates on magnetic and electric fields on rectilinear grids and user-provided initial conditions.

.. toctree::
   :maxdepth: 2

   whatsnew/index
   user-guide/index
   dev-guide/index
   api

Install DISCO
-------------
To install ccsdspy from PyPI, you can use the following command. If you are not using CUDA v12, replace `cupy-cuda12x` with `cupy` (this will build CuPy from source).

.. code::

   pip install disco cupy-cuda12x


Brief Tour
----------
This section is under construction.

.. code-block:: python

    import disco
    from disco.readers import SwmfOutFieldModelDataset

    dataset = SwmfOutFieldModelDataset('swmf_run/*.out')

    field_model_loader = disco.LazyFieldModelLoader(
        dataset, config, particle_state.mass, particle_state.charge
    )

    history = disco.trace_trajectory(
	config, particle_state, field_model_loader
    )
    history.save('DiscoTrajectoryOutput.h5')
    history.plot_xz()
    plt.savefig('xz_plot.png')

