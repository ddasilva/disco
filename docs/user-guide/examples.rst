.. _examples:

#########
Examples
#########

.. contents::
   :depth: 3

Performing Traces
*****************

Setting up Initial Conditions
===========================
Initial conditions are configured using the `~disco.ParticleState` class. In the below example, we setup one particle starting at (6.6, 0, 0) Earth Radii. To configure multiple particles, arrays of longer length can be provided. 

Because DISCO uses the mass and charge of the particles to dimensionalize the simulation, it can only do traces for one combination of mass and charge at a time. 
Unless otherwise specified, all coordinates used in DISCO are Solar Magnetic (SM).

.. code-block:: python

    import disco
    from astropy import constants, units
    import numpy as np

    particle_state = disco.ParticleState(
        x = np.array([6.6]) * constants.R_earth,
	y = np.array([0.0]) * constants.R_earth,
	z = np.array([0.0]) * constants.R_earth,
	ppar = np.array([0.8]) * constants.c,
	magnetic_moment = np.array([800]) * units.MeV / units.G,
	mass = constants.m_e,
	charge = constants.e.si,
    )
	

Loading Simulation Output
===========================
This section under construction!

Starting the Trace
===================
This section under construction!


Saving and Plotting Results
******************************

Saving and Loading from Disk
=============================
This section under construction!


Built-in Plotting Methods
=============================
This section under construction!


Advanced Options
*****************

Backwards-Time Integration
===========================
This section under construction!

Tracing in non-Time-Dependent Fields 
==========================================
This section under construction!

Loading from Custom Simulation Output
===============================
This section under construction!
