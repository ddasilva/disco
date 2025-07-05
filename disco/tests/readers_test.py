"Tests for the disco.readers module" ""
from datetime import timedelta
import os
import tempfile

from astropy import units
import numpy as np
import pytest

from disco import FieldModel
from disco.readers import SwmfOutFieldModelDataset
from disco.tests import testing_utils


def create_swmf_directory(swmf_files):
    """Create a temporary directory for SWMF test files."""
    temp_dir = tempfile.TemporaryDirectory()

    # Write files with dummy content
    for swmf_file in swmf_files:
        swmf_file_path = os.path.join(temp_dir.name, swmf_file)

        with open(swmf_file_path, "w") as fh:
            fh.write("This is a dummy SWMF out file for testing purposes.")

    return temp_dir, swmf_files


def create_swmf_directory_relative_times():
    """Create a temporary directory for SWMF test files with relative timestamps."""
    # List of dummy SWMF out files to create
    return create_swmf_directory(
        [
            "3d__var_4_t00030600_n00022500.out",
            "3d__var_4_t00030700_n00022548.out",
            "3d__var_4_t00030800_n00022601.out",
            "3d__var_4_t00030900_n00022655.out",
        ]
    )


def create_swmf_directory_absolute_times():
    """Create a temporary directory for SWMF test files with absolute timestamps."""
    return create_swmf_directory(
        [
            "3d__var_1_e19971010-170000-000.out",
            "3d__var_1_e19971010-170400-000.out",
        ]
    )


def test_swmf_out_dataset_length():
    """Test that the length of the SWMF dataset matches the number of files."""
    temp_dir, swmf_files = create_swmf_directory_relative_times()
    dataset = SwmfOutFieldModelDataset(f"{temp_dir.name}/*.out")

    assert len(dataset) == len(swmf_files), "Dataset length should match number of .out files"


def do_swmf_out_dataset_time_test(temp_dir, swmf_files, expected_times):
    dataset = SwmfOutFieldModelDataset(f"{temp_dir.name}/*.out")

    got_times = dataset.get_time_axis()

    # Check that the time axis has the correct length
    assert len(got_times) == len(swmf_files), "Time axis length should match dataset length"

    # Check that the time axis is in seconds
    assert got_times.unit.is_equivalent(units.s), "Time axis should have units of seconds"

    # Check that the time axis is what is expected
    assert np.all(got_times == expected_times), "Time axis values should match expected values"

    # Check that the first timestamp is zero
    assert dataset.get_time_axis()[0] == 0 * units.s, "First timestamp should be zero"


def test_swmf_out_dataset_time_axis_relative():
    """Test that the time axis is correctly populated from the filenames."""
    temp_dir, swmf_files = create_swmf_directory_relative_times()
    expected_times = np.array([i for i in range(len(swmf_files))]) * units.minute
    do_swmf_out_dataset_time_test(temp_dir, swmf_files, expected_times)


def test_swmf_out_dataset_time_axis_absolute():
    """Test that the time axis is correctly populated from the filenames."""
    temp_dir, swmf_files = create_swmf_directory_absolute_times()
    expected_times = np.array([0, 240]) * units.s
    do_swmf_out_dataset_time_test(temp_dir, swmf_files, expected_times)


@pytest.mark.slow
def test_swmf_out_dataset_cache_dir():
    """Test that the cache directory is created and used correctly."""
    temp_dir = tempfile.TemporaryDirectory()
    swmf_out_file = testing_utils.get_swmf_out_file()

    # Use a very downsampled grid for testing to make tests go faster
    dataset = SwmfOutFieldModelDataset(
        swmf_out_file, cache_regrid=True, grid_downsample=8, cache_regrid_dir=temp_dir.name
    )

    cache_file = dataset.get_cache_file(0)
    dataset[0]
    assert os.path.exists(cache_file), "Cache file should be created"
    assert len(os.listdir(temp_dir.name)) >= 0, "Cache directory should not be empty"


@pytest.mark.slow
def test_swmf_out_getitem():
    """Tests loading a SWMF .out file using the Dataset's __getitem__ method."""
    swmf_out_file = testing_utils.get_swmf_out_file()

    # Use a very downsampled grid for testing to make tests go faster
    dataset = SwmfOutFieldModelDataset(
        swmf_out_file, grid_downsample=8, cache_regrid=False, verbose=0
    )

    # Get first item
    field_model = dataset[0]

    assert isinstance(field_model, FieldModel), "Should return a FieldModel instance"
    assert field_model.axes.t.size == 1
    assert field_model.axes.t == 0 * units.s, "Time axis should be zero for first item"
    assert field_model.Bx.shape == (64, 19, 19, 1)

    # Get standard deviations of B field components for regridding regressions
    B_mean_got = [
        np.abs(field_model.Bx).mean(),
        np.abs(field_model.By).mean(),
        np.abs(field_model.Bz).mean(),
    ]
    B_mean_expected = [8.66971071 * units.nT, 18.8103232 * units.nT, 4.57709881 * units.nT]
    B_mean_threshold = 1e-2 * units.nT

    for got, expected in zip(B_mean_got, B_mean_expected):
        assert (
            np.abs(got - expected) < B_mean_threshold
        ), f"Field model B field abs mean should match expected value: {got} != {expected}"

    # Get standard deviations of E field components for regridding regressions
    E_mean_got = [
        np.abs(field_model.Ex).mean(),
        np.abs(field_model.Ey).mean(),
        np.abs(field_model.Ez).mean(),
    ]
    unit = units.mV / units.m
    E_mean_expected = [0.34928055 * unit, 0.39637236 * unit, 5.35730062 * unit]
    E_mean_threshold = 1e-2 * unit

    for got, expected in zip(E_mean_got, E_mean_expected):
        assert (
            np.abs(got - expected) < E_mean_threshold
        ), f"Field model E field abs mean should match expected value: {got} != {expected}"


def test_swmf_out_dataset_invalid_glob():
    """Test that an error is raised if the glob pattern does not match any out files."""
    # Create a dataset with a glob pattern that matches no files
    with pytest.raises(FileNotFoundError):
        SwmfOutFieldModelDataset("/nonexistance/*.out")


def test_swmf_out_dataset_t0():
    """Test that the t0 parameter is correctly applied to the time axis."""
    temp_dir, swmf_files = create_swmf_directory_relative_times()
    t0 = 3 * units.hour + 8 * units.minute
    dataset = SwmfOutFieldModelDataset(f"{temp_dir.name}/*.out", t0=t0)

    got_times = dataset.get_time_axis()
    expected_times = (
        3 * units.hour + np.array([(6 + i) for i in range(len(swmf_files))]) * units.minute
    )
    expected_times -= t0
    expected_times = expected_times.to(units.s)

    assert units.isclose(
        expected_times, got_times
    ).all(), "Time axis values should match expected values"
