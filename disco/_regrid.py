"""Regridding tools for DISCO"""
import os

from astropy import constants, units
from cupyx.scipy.spatial import KDTree
import cupy as cp
import numpy as np
import pandas as pd

from disco._axes import Axes
from disco._field_model import FieldModel
from disco.constants import DEFAULT_B0

# Location of target grid on disk and associated inner boundary for it
CUR_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
TARGET_GRID = os.path.join(CUR_DIRECTORY, "data/OpenGGCMGrids/overview_7M_now_11.8Mcells/")
TARGET_RINNER = 2.5 * constants.R_earth


def regrid_pointcloud(
    x_pc, y_pc, z_pc, t, Bx_pc, By_pc, Bz_pc, Ex_pc, Ey_pc, Ez_pc, k=8, B0=DEFAULT_B0
):
    """
    Regrid pointcloud data using radial basis functions and k-NN.

    Uses an OpenGGCM grid with 11.8M cells.

    Parameters
    ----------
    x_pc, y_pc, z_pc: array with no units
        Pointc loud XYZ positions in Re
    t: float
        Time position in seconds, used to assign time axis
    Bx_pc, By_pc, Bz_pc: array with no units
        External magnetic field in units in nT
    Ex_pc, Ey_pc, Ez_pc: array with no units
        Electric field in units of mV/m
    k: int
        Number of nearest neighbors to use.
    B0: scalar with units of magnetic field strength
        Internal field model strength passed to FieldModel()

    Returns
    -------
    FieldModel with one timestep set to the provided value of t
    """
    # Get new grid axis
    xaxis, yaxis, zaxis, taxis = get_new_grid(t)
    X, Y, Z = np.meshgrid(xaxis, yaxis, zaxis, indexing="ij")

    # Build KDTree from pointcloud and query at grid
    tree_points = cp.array([x_pc, y_pc, z_pc]).T
    tree = KDTree(tree_points)

    # Perform regridding by querying neighbors
    query_points = cp.array([X.flatten(), Y.flatten(), Z.flatten()]).T
    d, I = tree.query(query_points, k=k)

    # Calculate scale of radial basis function gaussians
    scale = d.mean(axis=1)
    scale_ = cp.zeros((scale.size, k))
    for i in range(k):
        scale_[:, i] = scale

    # Collect weighted average of neighbors
    regrid_data = {}
    target_shape = X.shape + (1,)
    vars = ["Bx", "By", "Bz", "Ex", "Ey", "Ez"]
    for var in vars:
        regrid_data[var] = np.zeros(target_shape)

    # Use Gaussian RBFs with scale apprximated by average neighbor distances
    weights = np.exp(-((d / scale_) ** 2))
    norm = weights.sum(axis=1)

    for i in range(k):
        weights[:, i] /= norm

    regrid_data["Bx"][:, :, :, 0] = (
        cp.sum(cp.array(Bx_pc)[I] * weights, axis=1).get().reshape(X.shape)
    )
    regrid_data["By"][:, :, :, 0] = (
        cp.sum(cp.array(By_pc)[I] * weights, axis=1).get().reshape(X.shape)
    )
    regrid_data["Bz"][:, :, :, 0] = (
        cp.sum(cp.array(Bz_pc)[I] * weights, axis=1).get().reshape(X.shape)
    )
    regrid_data["Ex"][:, :, :, 0] = (
        cp.sum(cp.array(Ex_pc)[I] * weights, axis=1).get().reshape(X.shape)
    )
    regrid_data["Ey"][:, :, :, 0] = (
        cp.sum(cp.array(Ey_pc)[I] * weights, axis=1).get().reshape(X.shape)
    )
    regrid_data["Ez"][:, :, :, 0] = (
        cp.sum(cp.array(Ez_pc)[I] * weights, axis=1).get().reshape(X.shape)
    )

    # Create FieldModel instance
    axes = Axes(
        xaxis * constants.R_earth,
        yaxis * constants.R_earth,
        zaxis * constants.R_earth,
        taxis * units.s,
        r_inner=TARGET_RINNER,
    )
    field_model = FieldModel(
        regrid_data["Bx"] * units.nT,
        regrid_data["By"] * units.nT,
        regrid_data["Bz"] * units.nT,
        regrid_data["Ex"] * units.mV / units.m,
        regrid_data["Ey"] * units.mV / units.m,
        regrid_data["Ez"] * units.mV / units.m,
        axes,
        B0=B0,
    )

    return field_model


def get_new_grid(t):
    """Definds the new grid to regrid to, based on OpenGGCM's rectilinear grid.

    Parameters
    ----------
    t: float
        Current timestep in seconds, no astropy units

    Returns
    -------
    xaxis, yaxis, zaxis, taxis: 1D arrays with no units
    """
    # Use OpenGGCM Grid specified in TARGET_GRID
    dfs = {}

    for dim in "xyz":
        dfs[dim] = pd.read_csv(
            f"{TARGET_GRID}/grid{dim}.txt", sep="\\s+", names=[dim, "delta", "unused2"], skiprows=1
        )

    xaxis = -dfs["x"].x[::-1].values
    yaxis = dfs["y"].y.values
    zaxis = dfs["z"].z.values
    taxis = np.array([t])

    return xaxis, yaxis, zaxis, taxis
