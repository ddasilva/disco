from astropy import units, constants
import h5py

from disco._dimensionalization import (
    undim_time,
    undim_magnetic_field,
    undim_space,
    undim_energy,
    undim_momentum,
    dim_time,
    dim_magnetic_moment,
    dim_space,
    dim_energy,
    dim_momentum,
)

# Units for when variables are stored on disk
TIME_UNITS = units.s
SPACE_UNITS = constants.R_earth
MOMENTUM_UNITS = units.keV * units.s / units.m
MAGFIELD_UNITS = units.nT
MAGNETIC_MOMENT_UNITS = units.MeV / units.nT
ENERGY_UNITS = units.eV
MASS_UNITS = units.kg
CHARGE_UNITS = units.C


class ParticleHistory:
    """History of particle states.

    See also:
    * `disco.TraceConfig(output_freq=...)`: Controlling between how many
      iterations between particle state is saved.
    """

    def __init__(self, t, x, y, z, ppar, M, B, W, h, mass, charge):
        self.t = t.to(TIME_UNITS)
        self.x = x.to(SPACE_UNITS)
        self.y = y.to(SPACE_UNITS)
        self.z = z.to(SPACE_UNITS)
        self.ppar = ppar.to(MOMENTUM_UNITS)
        self.M = M.to(MAGNETIC_MOMENT_UNITS)
        self.B = B.to(MAGFIELD_UNITS)
        self.W = W.to(ENERGY_UNITS)
        self.h = h.to(TIME_UNITS)
        self.mass = mass.to(MASS_UNITS)
        self.charge = charge.to(CHARGE_UNITS)

    def save(self, hdf_path):
        """Save particle history to an HDF5 file.

        Args
          hdf_path: Path to the HDF5 file where the history will be saved.
        See also:
          `ParticleHistory.load`: Load particle history from an HDF5 file.
        """
        with h5py.File(hdf_path, "w") as hdf:
            hdf["t"] = self.t.to_value(TIME_UNITS)
            hdf["x"] = self.x.to_value(SPACE_UNITS)
            hdf["y"] = self.y.to_value(SPACE_UNITS)
            hdf["z"] = self.z.to_value(SPACE_UNITS)
            hdf["ppar"] = self.ppar.to_value(MOMENTUM_UNITS)
            hdf["M"] = self.M.to_value(MAGNETIC_MOMENT_UNITS)
            hdf["B"] = self.B.to_value(MAGFIELD_UNITS)
            hdf["W"] = self.W.to_value(ENERGY_UNITS)
            hdf["h"] = self.h.to_value(TIME_UNITS)
            hdf["mass"] = self.mass.to_value(MASS_UNITS)
            hdf["charge"] = self.charge.to_value(CHARGE_UNITS)

            hdf["t"].attrs["UNITS"] = TIME_UNITS.to_string()
            hdf["x"].attrs["UNITS"] = SPACE_UNITS.to_string()
            hdf["y"].attrs["UNITS"] = SPACE_UNITS.to_string()
            hdf["z"].attrs["UNITS"] = SPACE_UNITS.to_string()
            hdf["ppar"].attrs["UNITS"] = MOMENTUM_UNITS.to_string()
            hdf["M"].attrs["UNITS"] = MAGNETIC_MOMENT_UNITS.to_string()
            hdf["B"].attrs["UNITS"] = MAGFIELD_UNITS.to_string()
            hdf["W"].attrs["UNITS"] = ENERGY_UNITS.to_string()
            hdf["h"].attrs["UNITS"] = TIME_UNITS.to_string()
            hdf["mass"].attrs["UNITS"] = MASS_UNITS.to_string()
            hdf["charge"].attrs["UNITS"] = CHARGE_UNITS.to_string()

    @classmethod
    def load(cls, hdf_path):
        """Load particle history from an HDF5 file.

        Args
          hdf_path: Path to the HDF5 file from which the history will be loaded.
        Returns
          ParticleHistory: An instance of ParticleHistory containing the loaded data.
        See also:
          `ParticleHistory.save`: Save particle history to an HDF5 file.
        """
        with h5py.File(hdf_path, "r") as hdf:
            t = hdf["t"][:] * TIME_UNITS
            x = hdf["x"][:] * SPACE_UNITS
            y = hdf["y"][:] * SPACE_UNITS
            z = hdf["z"][:] * SPACE_UNITS
            ppar = hdf["ppar"][:] * MOMENTUM_UNITS
            M = hdf["M"][:] * MAGNETIC_MOMENT_UNITS
            B = hdf["B"][:] * MAGFIELD_UNITS
            W = hdf["W"][:] * ENERGY_UNITS
            h = hdf["h"][:] * TIME_UNITS
            mass = hdf["mass"][()] * MASS_UNITS
            charge = hdf["charge"][()] * CHARGE_UNITS

        return cls(t, x, y, z, ppar, M, B, W, h, mass, charge)
