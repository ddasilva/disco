from disco._dimensionalization import (
    undim_time,
    undim_magnetic_field,
    undim_space,
    undim_energy,
    undim_momentum,
)


class ParticleHistory:
    """History of particle states.

    See also:
    * `disco.TraceConfig(output_freq=...)`: Controlling between how many
      iterations between particle state is saved.
    """

    def __init__(self, t, x, y, z, ppar, B, W, h, mass, charge):
        self.t = undim_time(t)
        self.x = undim_space(x)
        self.y = undim_space(y)
        self.z = undim_space(z)
        self.ppar = undim_momentum(ppar, mass)
        self.B = undim_magnetic_field(B, mass, charge)
        self.W = undim_energy(W, mass)
        self.h = undim_time(h)
