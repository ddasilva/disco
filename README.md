DISCO - Drift Integration Simulation COde
-----------------------------------------

DISCO is a tool for magnetospheric particle trajectory modeling, otherwise known as particle tracing, on the GPU. It calculates the trajectories of particles using [guiding center theory](https://farside.ph.utexas.edu/teaching/plasma/lectures/node18.html) with full support for relativistic effects. The tool operates on magnetic and electric fields on uniform grids and user-provided initial conditions.

DISCO is part of the larger Magnetosphere Aurora Reconnection Boundary Layer Explorer (MARBLE) project, a planned implementation of "collisionless Hall MHD" (or "kinetic Hall MHD") for planetary magnetospheres that self-consistently models the propagation of electrons and field-aligned currents from magnetic reconnection sites to the ionosphere. DISCO can also be used standalone without the rest of the framework.


## Installation
To install disco as a python model, run the following command from this directory.
```
$ pip install .
```

If you are interested in running the tests as well, instead use this command:
```
$ pip install .[dev]
```

## Running the tests
To run the test, simply issue the following command from the top-level directory.

```
$ pytest -v
```

## Team
Funded by LWS Strategic Capabilities Grant

* Daniel da Silva (DISCO Lead), [daniel.e.dasilva@nasa.gov](mailto:daniel.e.dasilva@nasa.gov)
* John Dorelli (MARBLE PI)


