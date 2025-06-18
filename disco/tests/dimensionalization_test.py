"""Tests the disco._dimensionalization module."""

from astropy import units
from astropy.constants import m_e, e, R_earth
from disco._dimensionalization import (
    dim_momentum,
    undim_momentum,
    dim_space,
    undim_space,
    dim_time,
    undim_time,
    dim_magnetic_field,
    undim_magnetic_field,
    dim_electric_field,
    undim_electric_field,
    undim_energy,
    dim_energy,
)


def test_dim_undim_momentum_roundtrip():
    """Test that dim_momentum and undim_momentum are inverse operations."""
    val = 1.2 * units.keV * units.s / units.m
    mass = m_e
    dimmed = dim_momentum(val, mass)
    undimmed = undim_momentum(dimmed, mass)
    assert undimmed.unit.is_equivalent(units.keV * units.s / units.m)
    assert abs((undimmed - val).to_value(units.keV * units.s / units.m)) < 1e-10


def test_dim_undim_space_roundtrip():
    """Test that dim_space and undim_space are inverse operations."""
    val = 2.5 * R_earth
    dimmed = dim_space(val)
    undimmed = undim_space(dimmed)
    assert undimmed.unit.is_equivalent(R_earth.unit)
    assert abs((undimmed - val).to_value(R_earth.unit)) < 1e-10


def test_dim_undim_time_roundtrip():
    """Test that dim_time and undim_time are inverse operations."""
    val = 100 * units.s
    dimmed = dim_time(val)
    undimmed = undim_time(dimmed)
    assert undimmed.unit.is_equivalent(units.s)
    assert abs((undimmed - val).to_value(units.s)) < 1e-10


def test_dim_undim_magnetic_field_roundtrip():
    """Test that dim_magnetic_field and undim_magnetic_field are inverse operations."""
    val = 50 * units.nT
    mass = m_e
    charge = e.si
    dimmed = dim_magnetic_field(val, mass, charge)
    undimmed = undim_magnetic_field(dimmed, mass, charge)
    assert undimmed.unit.is_equivalent(units.nT)
    assert abs((undimmed - val).to_value(units.nT)) < 1e-10


def test_dim_undim_electric_field_roundtrip():
    """Test that dim_electric_field and undim_electric_field are inverse operations."""
    val = 1.5 * units.mV / units.m
    mass = m_e
    charge = e.si
    dimmed = dim_electric_field(val, mass, charge)
    undimmed = undim_electric_field(dimmed, mass, charge)
    assert undimmed.unit.is_equivalent(units.mV / units.m)
    assert abs((undimmed - val).to_value(units.mV / units.m)) < 1e-10


def test_dim_undim_energy_roundtrip():
    """Test that dim_energy and undim_energy are inverse operations."""
    val = 2.0 * units.keV
    mass = m_e
    dimmed = dim_energy(val, mass)
    undimmed = undim_energy(dimmed, mass)
    assert undimmed.unit.is_equivalent(units.keV)
    assert abs((undimmed - val).to_value(units.keV)) < 1e-10
