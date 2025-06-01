"""Functions relating to dimensionalization."""
from astropy import constants, units


def dim_momentum(val, mass):
    """Dimensionalize a momentum value.
    Args
      val: value with units
      mass: mass of particle with units
    Returns
      value in no units
    """
    sf = constants.c * mass

    return (val / sf).to(1).value


def undim_momentum(val, mass):
    """Redimensionalize a momentum value.

    Args
      val: value with units
      mass: mass of particle
    Returns
        value with units, set to keV s / m
    """
    sf = constants.c * mass
    out_units = units.keV * units.s / units.m

    return (val * sf).to(out_units)


def undim_energy(val, mass):
    """Redimensionalize an energy value.

    Args
      val: value with units
      mass: mass of particle with units
    Returns
      value with no units.
    """
    return (val * mass * constants.c**2).to(units.keV)


def dim_space(val):
    """Dimensionalize a space value.
    Args
      value: value with units
    Returns
        value in no units
    """
    return val.to(constants.R_earth).value


def undim_space(val):
    """Redimensionalize a space value.

    Args
      value: value with units
    Returns
      value in redimensionalized units
    """
    return val * constants.R_earth


def dim_magnetic_field(val, mass, charge):
    """Redimensionalize a magnetic field value.

    Args
      value: value with units
    Returns
      value in no units
    """
    sf = charge * constants.R_earth / (mass * constants.c**2)
    B_units = units.s / constants.R_earth

    return (sf * val).to(B_units).value


def dim_electric_field(val, mass, charge):
    """Redimensionalize an electric field value.

    Args
      value: value with units
    Returns
      value in no units
    """
    sf = charge * constants.R_earth / (mass * constants.c**2)

    return (sf * val).to(1).value


def undim_magnetic_field(val, mass, charge):
    """Undimensionalize a magnetic field value.

    Args
        val: dimensionalized value out with units
        charge: charge of particle with units
        mass: mass of particle with units
    Returns
        value with units, set to nT
    """
    sf = charge * constants.R_earth / (mass * constants.c**2)
    B_units = units.s / constants.R_earth
    out_units = units.nT

    return (val * B_units / sf).to(out_units)


def undim_electric_field(val, mass, charge):
    """Redimensionalize an electric field value.

    Args
      value: value with units
    Returns
      value in redimensionalized units
    """
    sf = charge * constants.R_earth / (mass * constants.c**2)
    out_units = units.mV / units.m

    return (val / sf).to(out_units)


def dim_time(val):
    """Redimensionalize a time value.

    Args
      value: value with units
    Returns
      value in no units
    """
    sf = constants.c / constants.R_earth
    return (sf * val).to(1).value


def undim_time(val):
    """Redimensionalize a time value.

    Args
      value: value in seconds
    Returns
      value with units
    """
    sf = constants.R_earth / constants.c
    return val * sf
