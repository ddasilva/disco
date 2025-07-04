"""Regridding tools for DISCO"""
import os

from cupyx.scipy.spatial import KDTree
import cupy as cp
import numpy as np
import pandas as pd

# Location of target grid on disk and associated inner boundary for it
CUR_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
TARGET_GRID = os.path.join(CUR_DIRECTORY, "data/OpenGGCMGrids/overview_7M_now_11.8Mcells/")


def regrid_pointcloud(
    x_pc, y_pc, z_pc, point_cloud_fields, k=8, grid_downsample=2,
):
    """
    Regrid pointcloud data using radial basis functions and k-NN.

    Uses downsampled version of the OpenGGCM grid

    Parameters
    ----------
    x_pc, y_pc, z_pc: array with no units
        Pointc loud XYZ positions in earth radii
    point_cloud_fields: dictionary, string keys and array values
        Dictionary of fields to regrid. Must have same dimension as x_pc, y_pc, z_pc.
    k: int
        Number of nearest neighbors to use.
    grid_downsample: int
        Downsampling factor for the OpenGGCM grid. Every `grid_downsample`-th point is used.
        Default grid has 11.8 M cells.
    Returns
    -------
    xaxis, yaxis, zaxis: 1D arrays with no units
    regrid_fields: dictionary with same keys as point_cloud_fields
    """
    # Get new grid axis
    xaxis, yaxis, zaxis = get_new_grid(grid_downsample)
    X, Y, Z = np.meshgrid(xaxis, yaxis, zaxis, indexing="ij")

    # Build KDTree from pointcloud and query at grid
    tree_points = cp.array([x_pc, y_pc, z_pc]).T
    query_points = cp.array([X.flatten(), Y.flatten(), Z.flatten()]).T
    
    tree = KDTree(tree_points)
    d, I = tree.query(query_points, k=k)

    # Calculate scale of radial basis function gaussians
    scale = d.mean(axis=1)
    scale_ = cp.zeros((scale.size, k))
    for i in range(k):
        scale_[:, i] = scale

    # Collect weighted average of neighbors
    regrid_fields = {}
    target_shape = X.shape + (1,)
    for key in point_cloud_fields.keys():
        regrid_fields[key] = np.zeros(target_shape) * np.nan

    # Use Gaussian RBFs with scale apprximated by average neighbor distances
    weights = np.exp(-((d / scale_) ** 2))
    norm = weights.sum(axis=1)

    for i in range(k):
        weights[:, i] /= norm

    for key, np_input_array in point_cloud_fields.items():
        cp_input_array = cp.array(np_input_array)
        weighted_sum = cp.sum(cp_input_array[I] * weights, axis=1)
        regrid_fields[key][:, :, :, 0] = weighted_sum.get().reshape(X.shape)

    return xaxis, yaxis, zaxis, regrid_fields


def get_new_grid(grid_downsample):
    """Definds the new grid to regrid to, based on OpenGGCM's rectilinear grid.

    Parameters
    ----------
    grid_downsample: int
        Downsampling factor for the grid. Every `grid_downsample`-th point is used.
    
    Returns
    -------
    xaxis, yaxis, zaxis, 1D arrays with units
    """
    # Use OpenGGCM Grid specified in TARGET_GRID
    dfs = {}

    for dim in "xyz":
        dfs[dim] = pd.read_csv(
            f"{TARGET_GRID}/grid{dim}.txt", sep="\\s+", names=[dim, "delta", "unused2"], skiprows=1
        )

    xaxis = -dfs["x"].x[::-1].values[::grid_downsample]
    yaxis = dfs["y"].y.values[::grid_downsample]
    zaxis = dfs["z"].z.values[::grid_downsample]

    return xaxis, yaxis, zaxis
