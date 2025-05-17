class ParticleHistory:
    """History of particle states.

    See also:
    * `disco.TraceConfig(output_freq=...)`: Controlling between how many
      iterations between particle state is saved.
    """

    def __init__(self, t, x, y, z, ppar, B, W, h):
        self.t = t
        self.x = x
        self.y = y
        self.z = z
        self.ppar = ppar
        self.B = B
        self.W = W
        self.h = h
