"""Utilities to support testing DISCO functionality."""

import os
import requests


def get_file(path):
    """Downloads a testing file if not already on disk"""
    url = f"https://danieldasilva.org/ci_files/disco/{path}"

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(cur_dir, "data", path)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if not os.path.exists(path):
        resp = requests.get(url)
        with open(out_path, "wb") as fh:
            fh.write(resp.content)

    return out_path


def get_swmf_cdf_file():
    """Gets a single timestep of SWMF CDF data for testing."""
    return get_file("swmf_test_data/3d__var_1_e20151202-050000-000.out.cdf")
