
DISCO: Drift Orbit Simulation Code
==================================

.. warning::
   Warning: this code is under active development, and the API will change. Please check back later or contact the author if you plan to use this code.


DISCO is a tool for magnetospheric particle trajectory modeling, also known as particle tracing, on the GPU. It calculates the trajectories of particles using guiding center theory with full support for relativistic effects. The tool operates on magnetic and electric fields on rectilinear grids and user-provided initial conditions.

.. toctree::
   :maxdepth: 2

   whatsnew/index
   user-guide/index
   dev-guide/index
   api

Install DISCO
-------------
To install disco from PyPI, you can use the following command. If you are not using CUDA v12, replace `cupy-cuda12x` with `cupy` (this will build CuPy from source).

.. code::

   pip install disco-magnetosphere cupy-cuda12x


Brief Tour
----------

.. code-block:: python

   import disco
   from disco.readers import SwmfOutFieldModelDataset
   from astropy import units

   # Set initial conditions and dataset  ---------------------
   particle_state = disco.ParticleState(
      x, y, z, ppar, magnetic_moment, mass, charge
   )
   dataset = SwmfOutFieldModelDataset('swmf_run/*.out')

   field_model_loader = disco.LazyFieldModelLoader(
      dataset, config, mass, charge
   )

   config = disco.TraceConfig(t_final = 30 * units.s, output_freq = 1)

   # Do particle trace ---------------------------------------
   history = disco.trace_trajectory(
      config, particle_state, field_model_loader
   )

   history.save('DiscoTrajectoryOutput.h5')
   
   # Plot Results --------------------------------------------
   history.plot_xz()
   plt.savefig('xz_plot.png')

