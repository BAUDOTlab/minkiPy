
#!/usr/bin/env python
from . import io

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import numpy as np
import pandas as pd
import scanpy as sc
import json
from scipy.ndimage import label
import numba
from numba import njit
from mpi4py import MPI
import math
import glob  
import gc
import sys
import tempfile
import subprocess
import argparse


# MPI communicator and rank information, used to distribute genes across processes.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import sys
def build_grid(xmin, grid_size, resolution, dtype=np.float32):
    """
    Construct a 1D array of grid cell edges for a square, regular lattice.

    Parameters
    ----------
    xmin : float
        Left-most coordinate of the grid in physical units.
    grid_size : int
        Number of cells along each dimension.
    resolution : float
        Cell size in physical units (edge-to-edge spacing).
    dtype : numpy.dtype, optional
        Data type of the returned array (default: np.float32).

    Returns
    -------
    edges : ndarray of shape (grid_size + 1,)
        Monotonically increasing coordinates of cell edges. Cell centres can
        be reconstructed on-the-fly as (i + 0.5) * resolution + xmin.
    """
    return np.linspace(
        xmin,
        xmin + grid_size * resolution,
        grid_size + 1,
        dtype=dtype,
    )


@njit(boundscheck=False, cache=True, nogil=True, fastmath=True)
def Gaussian_Smoothing(x_locations, y_locations, edges):
    """
    Compute a Gaussian-smoothed 2D density field on a regular grid.

    Parameters
    ----------
    x_locations, y_locations : 1D arrays of float
        Coordinates of individual transcripts (or particles) in physical units.
    edges : 1D array of float
        Grid cell edges as returned by ``build_grid``. The grid is assumed
        square and the same edges are used for x and y.

    Notes
    -----
    The kernel is an isotropic Gaussian with standard deviation
    sigma = 2 * (edges[1] - edges[0]). Contributions are truncated to a
    compact support of ±2 sigma around each point to limit the cost.
    The implementation avoids constructing temporary meshgrids.

    Returns
    -------
    density : 2D array of float32, shape (n_bins, n_bins)
        Gaussian-smoothed density evaluated at grid cell centres.
    """
    reso   = edges[1] - edges[0]
    sigma  = 2.0 * reso
    two_sigma        = 2.0 * sigma
    inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma)

    # 1-D array of cell centres (physical coordinates)
    centres = (edges[:-1] + edges[1:]) * 0.5
    n_bins  = len(centres)

    # Target density grid
    density = np.zeros((n_bins, n_bins), dtype=np.float32)

    # Loop over points and update only the local neighbourhood within ±2σ
    for idx in range(len(x_locations)):
        x_pt = x_locations[idx]
        y_pt = y_locations[idx]

        # Bounding box [pt ± 2σ] in coordinate space
        x_min = x_pt - two_sigma
        x_max = x_pt + two_sigma
        y_min = y_pt - two_sigma
        y_max = y_pt + two_sigma

        # Convert to index ranges in the centres array
        i_low  = np.searchsorted(centres, x_min, side='left')
        i_high = np.searchsorted(centres, x_max, side='right')
        j_low  = np.searchsorted(centres, y_min, side='left')
        j_high = np.searchsorted(centres, y_max, side='right')

        if i_low >= i_high or j_low >= j_high:
            continue

        # Local update on the bounding sub-grid
        for i in range(i_low, i_high):
            dx = centres[i] - x_pt
            dx2 = dx * dx
            for j in range(j_low, j_high):
                dy = centres[j] - y_pt
                density[i, j] += math.exp(-(dx2 + dy * dy) * inv_two_sigma_sq)

    return density




def FoF_on_binary(binary_field):
    """
    Label connected components in a 2D binary mask using 8-connectivity.

    Parameters
    ----------
    binary_field : 2D array of int or bool
        Binary map where non-zero pixels are considered "inside" the excursion set.

    Returns
    -------
    labeled_array : 2D array of int
        Array of the same shape as ``binary_field`` where each connected component
        has a unique positive integer label and background is 0.
    """
    # 3x3 ones => 8-connected neighbourhood (including diagonals)
    structure = np.ones((3, 3), dtype=np.int32)
    labeled_array, num_features = label(binary_field, structure=structure)
    return labeled_array

@njit(boundscheck=False, cache=True, nogil=True)
def level_set_filtering(nu, level_set):
    """
    Threshold a continuous field at a given level to form a binary excursion set.

    Parameters
    ----------
    nu : 2D array of float
        Continuous scalar field (e.g. normalised density).
    level_set : float
        Threshold level. Cells with nu >= level_set are marked as 'inside'.

    Returns
    -------
    binary_field : 2D array of int32
        Binary mask where 1 indicates nu >= level_set and 0 otherwise.
    """
    binary_field = (nu >= level_set).astype(np.int32)
    return binary_field

@njit(boundscheck=False, cache=True, nogil=True)
def linear_interp(level_set, delta_a, delta_b):
    """
    Linearly interpolate the position where a scalar field crosses a level.

    Parameters
    ----------
    level_set : float
        Target level between the two corner values.
    delta_a, delta_b : float
        Field values at the two endpoints of an edge.

    Returns
    -------
    t : float
        Interpolation fraction in [0, 1] such that
        value(t) = (1 - t) * delta_a + t * delta_b = level_set.
        For perfectly flat edges (delta_a == delta_b), returns 0.5.
    """
    # Prevent division by zero on perfectly flat edges
    if delta_b == delta_a:      # rare but can happen
        return 0.5              # midpoint of the edge
    return (level_set - delta_a) / (delta_b - delta_a)

@njit(boundscheck=False, cache=True, nogil=True)
def marching_squares_algo(binary_field, labeled_field, level_set, nu):
    """
    Marching-squares contour extraction and area accounting for excursion sets.

    Parameters
    ----------
    binary_field : 2D array of int
        Binary mask (0/1) defining the excursion set at the given level.
    labeled_field : 2D array of int
        Connected-component labels for ``binary_field`` (e.g. output of FoF_on_binary).
    level_set : float
        Threshold level used to build ``binary_field`` (needed for interpolation).
    nu : 2D array of float
        Original continuous field from which the excursion set was derived.

    Returns
    -------
    contour_x_d, contour_y_d : 1D arrays of float32
        Start coordinates (x,y) of each boundary segment.
    contour_x_f, contour_y_f : 1D arrays of float32
        End coordinates (x,y) of each boundary segment.
    area_CR : 1D array of float64
        Accumulated area per region label; index 0 stores the total excursion-set area.
    AR : 1D array of int16
        Region label associated with each contour segment (or -1 if no region).
    """
    n = binary_field.shape[0]
    m = binary_field.shape[1]

    # ---------- Pass 0 : estimate how many contour segments we will need ----------
    seg_estimate = 0
    for i in range(n - 1):
        for j in range(m - 1):
            config = (binary_field[i,   j+1] |
                     (binary_field[i+1, j+1] << 1) |
                     (binary_field[i+1, j  ] << 2) |
                     (binary_field[i,   j  ] << 3)) + 1
            if config in (1, 16):  # no boundary
                continue
            seg_estimate += 1 if config not in (6, 11) else 2

    # Pre-allocate arrays for contour segments and region labels
    contour_x_d = np.empty(seg_estimate, dtype=np.float32)
    contour_y_d = np.empty(seg_estimate, dtype=np.float32)
    contour_x_f = np.empty(seg_estimate, dtype=np.float32)
    contour_y_f = np.empty(seg_estimate, dtype=np.float32)
    AR          = np.empty(seg_estimate, dtype=np.int16)

    max_label = labeled_field.max()
    # area_CR[0] will store the total 'in-region' area, area_CR[r] the area of region r
    area_CR   = np.zeros(max_label + 1, dtype=np.float64)

    # Number of segments actually filled so far
    k = 0

    for i in range(n - 1):
        for j in range(m - 1):
            # Identify the 4 corners of this cell
            c0  = binary_field[i,   j+1]   # top-left
            c0v = nu[i,   j+1]
            c1  = binary_field[i+1, j+1]   # top-right
            c1v = nu[i+1, j+1]
            c2  = binary_field[i+1, j]     # bottom-right
            c2v = nu[i+1, j]
            c3  = binary_field[i,   j]     # bottom-left
            c3v = nu[i,   j]

            # Build config in [1..16]
            config = (c0 | (c1 << 1) | (c2 << 2) | (c3 << 3)) + 1

            # Gather region labels for these corners:
            r0 = labeled_field[i,   j+1]
            r1 = labeled_field[i+1, j+1]
            r2 = labeled_field[i+1, j  ]
            r3 = labeled_field[i,   j  ]

            # Pick the smallest non-zero label among the four corners (0 = background)
            reg = 0
            if r0 != 0:
                reg = r0
            if r1 != 0 and (reg == 0 or r1 < reg):
                reg = r1
            if r2 != 0 and (reg == 0 or r2 < reg):
                reg = r2
            if r3 != 0 and (reg == 0 or r3 < reg):
                reg = r3

            if config == 16:
                # All corners are 'in', so the full pixel is inside a region
                if reg != 0:
                    area_CR[0]   += 1.0
                    area_CR[reg] += 1.0
                continue

            if config == 1:
                # All corners are out => no boundary, no area
                continue

            # Otherwise, we have one or two boundary segments to define.
            # We work in cell-centre coordinates (i + 0.5, j + 0.5).
            c0x = i   + 0.5 ; c1x = i+1 + 0.5
            c2x = i+1 + 0.5 ; c3x = i   + 0.5

            c0y = j+1 + 0.5 ; c1y = j+1 + 0.5
            c2y = j   + 0.5 ; c3y = j   + 0.5

            # Local holders for up to two segments:
            # segment #1: (dx1, dy1) -> (fx1, fy1)
            # segment #2: (dx2, dy2) -> (fx2, fy2)
            seg_count = 0
            dx1 = dy1 = fx1 = fy1 = 0.0
            dx2 = dy2 = fx2 = fy2 = 0.0

            # Area contribution from this cell to the excursion set
            pix_area = 0.0

            # Case-by-case handling of the 16 marching-squares configurations
            if config == 2:
                c0c3 = linear_interp(level_set, c0v, c3v)
                c0c1 = linear_interp(level_set, c0v, c1v)
                dx1 = c0x        ; dy1 = c0y - c0c3
                fx1 = c0x + c0c1 ; fy1 = c0y
                seg_count = 1
                pix_area = (c0c3 * c0c1) / 2.0

            elif config == 3:
                c1c0 = linear_interp(level_set, c1v, c0v)
                c1c2 = linear_interp(level_set, c1v, c2v)
                dx1 = c1x - c1c0 ; dy1 = c1y
                fx1 = c1x        ; fy1 = c1y - c1c2
                seg_count = 1
                pix_area = (c1c0 * c1c2) / 2.0

            elif config == 4:
                c0c3 = linear_interp(level_set, c0v, c3v)
                c1c2 = linear_interp(level_set, c1v, c2v)
                dx1 = c0x ; dy1 = c0y - c0c3
                fx1 = c1x ; fy1 = c1y - c1c2
                seg_count = 1
                width = abs(c1x - c0x)
                height1 = min(c0c3, c1c2)
                height2 = abs(c0c3 - c1c2) / 2.0
                pix_area = width * (height1 + height2)

            elif config == 5:
                c2c1 = linear_interp(level_set, c2v, c1v)
                c2c3 = linear_interp(level_set, c2v, c3v)
                dx1 = c2x        ; dy1 = c2y + c2c1
                fx1 = c2x - c2c3 ; fy1 = c2y
                seg_count = 1
                pix_area = (c2c1 * c2c3) / 2.0

            elif config == 6:
                # Two separate boundary segments
                c0c3 = linear_interp(level_set, c0v, c3v)
                c2c3 = linear_interp(level_set, c2v, c3v)
                c2c1 = linear_interp(level_set, c2v, c1v)
                c0c1 = linear_interp(level_set, c0v, c1v)

                # 1st segment
                dx1 = c0x   ; dy1 = c0y - c0c3
                fx1 = c2x   ; fy1 = c2y + c2c1

                # 2nd segment
                dx2 = c2x - c2c3 ; dy2 = c2y
                fx2 = c0x + c0c1 ; fy2 = c0y

                seg_count = 2
                pix_area = 1.0 - (
                    (1.0 - c0c3) * (1.0 - c2c3) +
                    (1.0 - c2c1) * (1.0 - c0c1)
                )

            elif config == 7:
                c1c0 = linear_interp(level_set, c1v, c0v)
                c2c3 = linear_interp(level_set, c2v, c3v)
                dx1 = c1x - c1c0 ; dy1 = c1y
                fx1 = c2x - c2c3 ; fy1 = c2y
                seg_count = 1
                h = abs(c2y - c1y)
                a = min(c1c0, c2c3)
                b = abs(c1c0 - c2c3) / 2.0
                pix_area = h * (a + b)

            elif config == 8:
                c0c3 = linear_interp(level_set, c0v, c3v)
                c2c3 = linear_interp(level_set, c2v, c3v)
                dx1 = c0x        ; dy1 = c0y - c0c3
                fx1 = c2x - c2c3 ; fy1 = c2y
                seg_count = 1
                # Explicit parentheses to avoid ambiguity in the area expression
                pix_area = 1.0 - (1.0 - c0c3 * (1.0 - c2c3))

            elif config == 9:
                c3c2 = linear_interp(level_set, c3v, c2v)
                c3c0 = linear_interp(level_set, c3v, c0v)
                dx1 = c3x + c3c2 ; dy1 = c3y
                fx1 = c3x        ; fy1 = c3y + c3c0
                seg_count = 1
                pix_area = (c3c2 * c3c0) / 2.0

            elif config == 10:
                c3c2 = linear_interp(level_set, c3v, c2v)
                c0c1 = linear_interp(level_set, c0v, c1v)
                dx1 = c3x + c3c2 ; dy1 = c3y
                fx1 = c0x + c0c1 ; fy1 = c0y
                seg_count = 1
                h = abs(c3y - c0y)
                a = min(c3c2, c0c1)
                b = abs(c3c2 - c0c1) / 2.0
                pix_area = h * (a + b)

            elif config == 11:
                # Two separate boundary segments
                c1c0 = linear_interp(level_set, c1v, c0v)
                c3c0 = linear_interp(level_set, c3v, c0v)
                c3c2 = linear_interp(level_set, c3v, c2v)
                c1c2 = linear_interp(level_set, c1v, c2v)

                dx1 = c1x - c1c0 ; dy1 = c1y
                fx1 = c3x + c3c2 ; fy1 = c3y

                dx2 = c3x ; dy2 = c3y + c3c0
                fx2 = c1x ; fy2 = c1y - c1c2

                seg_count = 2
                pix_area = 1.0 - (
                    (1.0 - c1c0) * (1.0 - c3c0) +
                    (1.0 - c3c2) * (1.0 - c1c2)
                )

            elif config == 12:
                c3c2 = linear_interp(level_set, c3v, c2v)
                c1c2 = linear_interp(level_set, c1v, c2v)
                dx1 = c3x + c3c2 ; dy1 = c3y
                fx1 = c1x        ; fy1 = c1y - c1c2
                seg_count = 1
                pix_area = 1.0 - ((1.0 - c3c2) * (1.0 - c1c2))

            elif config == 13:
                c2c1 = linear_interp(level_set, c2v, c1v)
                c3c0 = linear_interp(level_set, c3v, c0v)
                dx1 = c2x ; dy1 = c2y + c2c1
                fx1 = c3x ; fy1 = c3y + c3c0
                seg_count = 1
                w = abs(c2x - c3x)
                a = min(c2c1, c3c0)
                b = abs(c2c1 - c3c0) / 2.0
                pix_area = w * (a + b)

            elif config == 14:
                c2c1 = linear_interp(level_set, c2v, c1v)
                c0c1 = linear_interp(level_set, c0v, c1v)
                dx1 = c2x        ; dy1 = c2y + c2c1
                fx1 = c0x + c0c1 ; fy1 = c0y
                seg_count = 1
                pix_area = 1.0 - ((1.0 - c2c1) * (1.0 - c0c1))

            elif config == 15:
                c1c0 = linear_interp(level_set, c1v, c0v)
                c3c0 = linear_interp(level_set, c3v, c0v)
                dx1 = c1x - c1c0 ; dy1 = c1y
                fx1 = c3x        ; fy1 = c3y + c3c0
                seg_count = 1
                pix_area = 1.0 - ((1.0 - c1c0) * (1.0 - c3c0))

            # Update area for [0, reg] if we found a non-zero region
            if reg != 0:
                area_CR[0]   += pix_area
                area_CR[reg] += pix_area

            # Insert the line segments into arrays (seg_count can be 1 or 2)
            if seg_count >= 1:
                contour_x_d[k] = dx1 ; contour_y_d[k] = dy1
                contour_x_f[k] = fx1 ; contour_y_f[k] = fy1
                AR[k] = reg if reg != 0 else -1
                k += 1

            if seg_count == 2:
                contour_x_d[k] = dx2 ; contour_y_d[k] = dy2
                contour_x_f[k] = fx2 ; contour_y_f[k] = fy2
                AR[k] = reg if reg != 0 else -1
                k += 1

    # Slice down to the used portion (k)
    contour_x_d = contour_x_d[:k] ; contour_y_d = contour_y_d[:k]
    contour_x_f = contour_x_f[:k] ; contour_y_f = contour_y_f[:k]
    AR          = AR[:k]

    return contour_x_d, contour_y_d, contour_x_f, contour_y_f, area_CR, AR

@njit
def pair_less(ax, ay, bx, by):
    """
    Lexicographic comparison between two 2D points.

    Parameters
    ----------
    ax, ay : float
        Coordinates of the first point (a_x, a_y).
    bx, by : float
        Coordinates of the second point (b_x, b_y).

    Returns
    -------
    bool
        True if (ax, ay) < (bx, by) in lexicographic order,
        i.e. ax < bx or (ax == bx and ay < by).
    """
    if ax < bx:
        return True
    elif ax > bx:
        return False
    else:
        return (ay < by)


@njit
def pair_equal(ax, ay, bx, by):
    """
    Test equality between two 2D points.

    Parameters
    ----------
    ax, ay : float
        Coordinates of the first point (a_x, a_y).
    bx, by : float
        Coordinates of the second point (b_x, b_y).

    Returns
    -------
    bool
        True if (ax, ay) and (bx, by) are exactly equal.
    """
    return (ax == bx) and (ay == by)


@njit
def find_equal_range(sorted_x, sorted_y, keyx, keyy):
    """
    Locate the contiguous block of entries equal to (keyx, keyy)
    in a lexicographically sorted list of 2D coordinates.

    The input (sorted_x, sorted_y) arrays are assumed to be sorted
    by (x, y) in lexicographic order. This function performs two
    binary searches to find the left- and right-most occurrences of
    (keyx, keyy).

    Parameters
    ----------
    sorted_x, sorted_y : 1D arrays of float
        Coordinates of points, sorted lexicographically by (x, y).
    keyx, keyy : float
        Coordinate pair we are searching for.

    Returns
    -------
    start_pos, end_pos : int
        Half-open index range [start_pos, end_pos) such that
        sorted_x[i] == keyx and sorted_y[i] == keyy for all
        i in [start_pos, end_pos). If no match is found, returns (0, 0).
    """
    N = len(sorted_x)

    # 1) Find left boundary (first index >= key)
    left = 0
    right = N
    while left < right:
        mid = (left + right) >> 1
        if pair_less(sorted_x[mid], sorted_y[mid], keyx, keyy):
            left = mid + 1
        else:
            right = mid
    start_pos = left

    # If out of bounds or not a match => no occurrences
    if start_pos >= N or not pair_equal(sorted_x[start_pos], sorted_y[start_pos], keyx, keyy):
        return 0, 0

    # 2) Find right boundary (first index > key)
    left = start_pos
    right = N
    while left < right:
        mid = (left + right) >> 1
        # We want the first mid where (sorted_x[mid], sorted_y[mid]) > (keyx, keyy)
        if pair_less(keyx, keyy, sorted_x[mid], sorted_y[mid]):
            right = mid
        else:
            left = mid + 1
    end_pos = left

    return start_pos, end_pos


def prepare_sorted_arrays(x_d, y_d, x_f, y_f):
    """
    Precompute lexicographically sorted views of segment endpoints.

    This helper builds two sets of arrays:
      * one for segment start points (x_d, y_d),
      * one for segment end points   (x_f, y_f),

    each accompanied by the corresponding original segment indices.
    These sorted arrays are later used for fast adjacency lookups.

    Parameters
    ----------
    x_d, y_d : 1D arrays of float
        Start (x, y) coordinates of segments.
    x_f, y_f : 1D arrays of float
        End   (x, y) coordinates of segments.

    Returns
    -------
    sstarts_x, sstarts_y : 1D arrays of float
        Start coordinates sorted lexicographically by (x, y).
    sstarts_idx : 1D array of int
        Indices mapping sorted starts back to the original segment indices.
    sends_x, sends_y : 1D arrays of float
        End coordinates sorted lexicographically by (x, y).
    sends_idx : 1D array of int
        Indices mapping sorted ends back to the original segment indices.
    """
    N = len(x_d)
    starts_xy = np.column_stack((x_d, y_d))
    ends_xy   = np.column_stack((x_f, y_f))
    idx_arr   = np.arange(N)

    # Sort by (x, y) using np.lexsort (last key is primary)
    sort_idx_starts = np.lexsort((starts_xy[:, 1], starts_xy[:, 0]))
    sort_idx_ends   = np.lexsort((ends_xy[:, 1],   ends_xy[:, 0]))

    sstarts_x   = starts_xy[sort_idx_starts, 0]
    sstarts_y   = starts_xy[sort_idx_starts, 1]
    sstarts_idx = idx_arr[sort_idx_starts]

    sends_x   = ends_xy[sort_idx_ends, 0]
    sends_y   = ends_xy[sort_idx_ends, 1]
    sends_idx = idx_arr[sort_idx_ends]

    return (sstarts_x, sstarts_y, sstarts_idx,
            sends_x,   sends_y,   sends_idx)


@njit
def norm_adj_vec_angle(v1, v2):
    """
    Compute the signed angle between two 2D vectors, after normalisation.

    Parameters
    ----------
    v1, v2 : 1D arrays of float, shape (2,)
        2D vectors.

    Returns
    -------
    angle : float
        Signed angle in radians between v1 and v2, in the range [-pi, pi].
        Computed using atan2(det, dot) for numerical robustness.
    """
    eps = 1e-12
    n1 = eps + np.sqrt(v1[0] * v1[0] + v1[1] * v1[1])
    n2 = eps + np.sqrt(v2[0] * v2[0] + v2[1] * v2[1])
    v1x = v1[0] / n1
    v1y = v1[1] / n1
    v2x = v2[0] / n2
    v2y = v2[1] / n2
    dot = v1x * v2x + v1y * v2y
    det = v1x * v2y - v1y * v2x
    return np.arctan2(det, dot)


@njit
def accumulate_angles_via_sort(
    x_d, y_d, x_f, y_f,
    AR,
    sstarts_x, sstarts_y, sstarts_idx,
    sends_x,   sends_y,   sends_idx,
    angles,
):
    """
    Accumulate turning angles between adjacent boundary segments.

    This routine replaces an O(N^2) adjacency search by using sorted
    arrays and binary search:

      1) For each segment i, we find all segments j whose start point
         coincides with the end point of i, i.e. start_j == end_i.
         For every such pair (i, j) with j > i, we add the angle from
         segment i to segment j.

      2) Symmetrically, we find all segments j whose end point
         coincides with the start point of i, i.e. end_j == start_i.
         For every such pair (i, j) with j > i, we add the angle from
         segment j to segment i.

    In both cases, the angle contribution is added to:
      * angles[0]       : total over the entire excursion set,
      * angles[ AR[i] ] : contribution associated with the region of
                          segment i (if AR[i] >= 0).

    Parameters
    ----------
    x_d, y_d : 1D arrays of float
        Start coordinates of segments.
    x_f, y_f : 1D arrays of float
        End coordinates of segments.
    AR : 1D array of int
        Region labels for each segment (as returned by marching_squares_algo).
    sstarts_x, sstarts_y, sstarts_idx :
        Sorted start-coordinates and corresponding indices (from prepare_sorted_arrays).
    sends_x, sends_y, sends_idx :
        Sorted end-coordinates and corresponding indices (from prepare_sorted_arrays).
    angles : 1D array of float
        Accumulator array; modified in place.

    Returns
    -------
    None
        Results are written in-place into ``angles``.
    """
    N = len(x_d)
    for i in range(N):
        # 1) Segments whose start coincides with the end of segment i
        ex, ey = x_f[i], y_f[i]
        left, right = find_equal_range(sstarts_x, sstarts_y, ex, ey)
        for pos in range(left, right):
            j = sstarts_idx[pos]
            if j > i:
                v1 = np.array([x_f[i] - x_d[i], y_f[i] - y_d[i]])
                v2 = np.array([x_f[j] - x_d[j], y_f[j] - y_d[j]])
                angle = norm_adj_vec_angle(v1, v2)
                angles[0] += angle
                angles[AR[i]] += angle

        # 2) Segments whose end coincides with the start of segment i
        sx, sy = x_d[i], y_d[i]
        left, right = find_equal_range(sends_x, sends_y, sx, sy)
        for pos in range(left, right):
            j = sends_idx[pos]
            if j > i:
                v1 = np.array([x_f[j] - x_d[j], y_f[j] - y_d[j]])
                v2 = np.array([x_f[i] - x_d[i], y_f[i] - y_d[i]])
                angle = norm_adj_vec_angle(v1, v2)
                angles[0] += angle
                angles[AR[i]] += angle



def compute_Minkowski_Tensors(
    x_d,
    y_d,
    x_f,
    y_f,
    area_CR,
    area_CR_LS0,
    AR,
    binary_field,
    labeled_field,
    frac_area,
):
    """
    Compute scalar Minkowski functionals (W0, W1, W2) and a global anisotropy
    measure (Beta_tot) for a single excursion set.

    Parameters
    ----------
    x_d, y_d : 1D arrays of float
        Start coordinates of boundary segments, as returned by
        ``marching_squares_algo``.
    x_f, y_f : 1D arrays of float
        End coordinates of boundary segments, as returned by
        ``marching_squares_algo``.
    area_CR : 1D array of float
        In-region area per region label for the current level set.
        ``area_CR[0]`` stores the total area of the excursion set.
    area_CR_LS0 : float
        Total excursion-set area at a reference level (typically the first
        level in the profile); used to normalise W0, W1, and W2 so that
        all levels are expressed in a common unit.
    AR : 1D array of int
        Region label associated with each boundary segment (or -1 for
        segments not associated with a labelled region).
    binary_field : 2D array of int
        Binary mask defining the excursion set at the current level.
    labeled_field : 2D array of int
        Connected-component labels for ``binary_field``.
    frac_area : float
        Kept for backward compatibility. The function now normalises by
        ``area_CR_LS0`` rather than by a global fraction; this argument
        is not used.

    Returns
    -------
    W0, W1, W2, Beta_tot : float
        Scalar Minkowski measures for the whole field (label 0):
        - W0 : area (normalised by area_CR_LS0),
        - W1 : boundary length (normalised by area_CR_LS0),
        - W2 : integrated curvature / Euler-characteristic-like term
               (normalised by area_CR_LS0),
        - Beta_tot : global shape anisotropy in [0, 1], derived from
                     the eigenvalues of the rank-2 tensor W2_11.
    """

    # --- Handle extreme cases -----------------------------------------------
    # Case 1: Only background (no pixels above threshold)
    if np.all(binary_field == 0):
        # No detected region → empty set with isotropic reference Beta_tot
        return 0.0, 0.0, 0.0, 0.5

    # Case 2: Entire field is 'in' the excursion set
    if np.all(binary_field == 1):
        # Full map active, but no boundary → W1=W2=0, isotropic Beta_tot
        return 1.0, 0.0, 0.0, 0.5

    region_names = np.unique(labeled_field)
    sizz = len(region_names)

    # W0: area functional (normalised by the area at the reference level)
    # area_CR[0] is the total excursion-set area at the current level
    # This is of paramount important here : we renormalize each Minkowski 
    # caracteristic by area_CR_LS0 AND NOT by the area of the whole sample, 
    # since each gene has its own spatial region of definition. This avoid 
    # large batch effect between samples.
    # it also forces W0 to start at 1
    W0 = area_CR / area_CR_LS0

    # Distances (lengths) of each boundary segment
    distances = np.sqrt((x_f - x_d) ** 2 + (y_f - y_d) ** 2)

    # W1: boundary length per region
    totallength_CR = np.zeros(sizz, dtype=np.float64)
    unzero_regions = region_names[region_names != 0]
    for RN in unzero_regions:
        totallength_CR[RN] = np.sum(distances[AR == RN])

    W1_array = np.zeros(sizz, dtype=np.float64)
    # Total boundary length over all segments
    W1_array[0] = np.sum(distances)
    for RN in unzero_regions:
        W1_array[RN] = totallength_CR[RN]

    # Normalise W1 by the area at the reference level so that units are
    # consistent across levels
    W1_array *= (1.0 / (4.0 * area_CR_LS0))

    # -------------------------------------------------------------------------
    # W2: curvature term from turning angles between adjacent segments
    # -------------------------------------------------------------------------

    # Prepare sorted arrays of segment endpoints (Python space)
    sstarts_x, sstarts_y, sstarts_idx, sends_x, sends_y, sends_idx = prepare_sorted_arrays(
        x_d, y_d, x_f, y_f
    )

    # Accumulate signed turning angles per region
    angles = np.zeros(sizz, dtype=np.float64)
    accumulate_angles_via_sort(
        x_d,
        y_d,
        x_f,
        y_f,
        AR,
        sstarts_x,
        sstarts_y,
        sstarts_idx,
        sends_x,
        sends_y,
        sends_idx,
        angles,
    )

    # W2 is proportional to the sum of turning angles, normalised by the
    # area at the reference level
    W2_array = angles / (2.0 * np.pi * area_CR_LS0)

    # -------------------------------------------------------------------------
    # W2_11: rank-2 tensor version of W2 and global anisotropy Beta_tot
    # -------------------------------------------------------------------------

    W2_11 = np.zeros((sizz, 2, 2), dtype=np.float64)

    # First, accumulate tensor contributions for the whole set (label 0)
    for i in range(len(x_d)):
        e = np.array([x_f[i] - x_d[i], y_f[i] - y_d[i]], dtype=np.float64)
        norm_e = max(np.linalg.norm(e), 1e-9)
        for ii in range(2):
            for jj in range(2):
                W2_11[0, ii, jj] += e[ii] * e[jj] / norm_e

    # Then, accumulate per-region contributions for non-zero labels
    for RN in unzero_regions:
        cond = (AR == RN)
        x_d_R = x_d[cond]
        y_d_R = y_d[cond]
        x_f_R = x_f[cond]
        y_f_R = y_f[cond]
        for i in range(len(x_d_R)):
            e = np.array(
                [x_f_R[i] - x_d_R[i], y_f_R[i] - y_d_R[i]],
                dtype=np.float64,
            )
            norm_e = max(np.linalg.norm(e), 1e-9)
            for ii in range(2):
                for jj in range(2):
                    W2_11[RN, ii, jj] += e[ii] * e[jj] / norm_e

    # Normalise tensor entries by the same factor as W2_array
    W2_11 *= (1.0 / (2.0 * np.pi * area_CR_LS0))

    # Global anisotropy Beta_tot from eigenvalues of W2_11 for the whole set
    eigen_V = np.linalg.eigvals(W2_11[0])
    lam1, lam2 = np.max(eigen_V), np.min(eigen_V)
    Beta_tot = lam2 / max(lam1, 1e-9)

    # Return scalar Minkowski functionals for the full excursion set
    return W0[0], W1_array[0], W2_array[0], Beta_tot


def compute_MT_distribs(origin_field, nbr, frac_area):
    """
    Compute Minkowski functionals/tensors over a 1D family of level sets.

    Parameters
    ----------
    origin_field : 2D ndarray of float
        Normalised continuous field (e.g. smoothed density) on a regular
        grid, typically rescaled to [0, 1].
    nbr : int
        Number of level sets (excluding the trivial min/max levels).
    frac_area : float
        Kept for backward compatibility. The current implementation
        normalises by ``area_CR_LS0`` instead and does not use this
        argument internally.

    Returns
    -------
    LS : 1D ndarray of float
        Array of level-set thresholds in (0, 1) used to build excursion sets.
    W0_dist, W1_dist, W2_dist, Beta_tot_dist : 1D ndarrays of float
        Minkowski profiles as a function of level:
        - W0_dist    : area functional,
        - W1_dist    : boundary length functional,
        - W2_dist    : curvature / Euler-characteristic-like functional,
        - Beta_tot_dist : global anisotropy.
    area_CR_LS0 : float
        Total excursion-set area at the first level in LS; used as
        a common normalisation factor across all levels.
    """
    # Uniform sampling of levels in (0, 1); we drop the trivial min and max
    LS = np.linspace(0, 1, nbr + 2)[1:-1]

    # Alternative choices (e.g. log_binning) can be enabled if needed:
    # LS = np.linspace(-0.5, 0.5, nbr)

    W0_dist, W1_dist, W2_dist, Beta_tot_dist = [[] for _ in range(4)]
    m = 0
    for values in LS:
        # Binary excursion set at this level
        binary_field = level_set_filtering(origin_field, values)
        labeled_field = FoF_on_binary(binary_field)

        # Marching-squares extraction of boundary segments
        x_d, y_d, x_f, y_f, area_CR, AR = marching_squares_algo(
            binary_field, labeled_field, values, origin_field
        )

        # Store the reference area at the first level
        if m == 0:
            area_CR_LS0 = area_CR[0]

        # Compute Minkowski functionals/tensors at this level
        W0, W1, W2, Beta_tot = compute_Minkowski_Tensors(
            x_d,
            y_d,
            x_f,
            y_f,
            area_CR,
            area_CR_LS0,
            AR,
            binary_field,
            labeled_field,
            frac_area,
        )
        W0_dist.append(W0)
        W1_dist.append(W1)
        W2_dist.append(W2)
        Beta_tot_dist.append(Beta_tot)
        m += 1

    return (
        np.array(LS),
        np.array(W0_dist),
        np.array(W1_dist),
        np.array(W2_dist),
        np.array(Beta_tot_dist),
        area_CR_LS0,
    )


def data_to_Mink(x_locations, y_locations, edges, nbr, frac_area):
    """
    Compute Minkowski profiles from a set of point coordinates.

    This is the high-level entry point: starting from a discrete set of
    (x, y) positions (e.g. transcripts for one gene), it builds a
    Gaussian-smoothed field on the grid, normalises it, and evaluates
    Minkowski functionals/tensors across a 1D family of level sets.

    Parameters
    ----------
    x_locations, y_locations : 1D arrays of float
        Coordinates of individual points (e.g. transcript positions) in
        physical units consistent with ``edges``.
    edges : 1D ndarray of float
        Grid cell edges as returned by ``build_grid``; the same edges are
        used for x and y directions.
    nbr : int
        Number of level sets used to build the Minkowski profiles.
    frac_area : float
        Kept for backward compatibility; not used in the current
        normalisation scheme (see ``compute_MT_distribs``).

    Returns
    -------
    LS : 1D ndarray of float
        Level-set thresholds in (0, 1) used to construct excursion sets.
    W0_dist, W1_dist, W2_dist, Beta_tot_dist : 1D ndarrays of float
        Minkowski profiles as a function of level.
    min_val, max_val : float
        Minimum and maximum values of the smoothed field before
        normalisation to [0, 1].
    area_LS0 : float
        Total excursion-set area at the first level in LS (same as
        ``area_CR_LS0`` from ``compute_MT_distribs``).
    """
    # Gaussian smoothing of the point cloud on the regular grid
    raw = Gaussian_Smoothing(x_locations, y_locations, edges)

    # Explicit pathological case: no transcript for this gene
    if len(x_locations) == 0:
        raise RuntimeError(
            "[data_to_Mink] Empty field for this gene (0 transcripts). "
            "Your gene filter is too permissive: increase MIN_COUNTS / MIN_SPOTS."
        )

    # Stabilise by the global mean to reduce scale variations
    mean_raw = float(raw.mean())

    # Strict test: if the mean is zero or non-finite, abort immediately
    if (not np.isfinite(mean_raw)) or (mean_raw <= 0.0):
        raise RuntimeError(
            f"[data_to_Mink] Smoothed field has a non-positive mean (mean_raw={mean_raw:.3g}). "
            "This gene has too few transcripts to be analysed reliably. "
            "Your gene filter is too permissive: increase MIN_COUNTS / MIN_SPOTS."
        )
     
    raw /= np.float32(mean_raw)

    # Normalise the field to [0, 1] (after min–max rescaling)
    min_val = raw.min()
    max_val = raw.max()

    if max_val > min_val:
        density = (raw - np.float32(min_val)) / np.float32(max_val - min_val)
    else:
        # Pathological case: completely flat field
        density = np.zeros_like(raw, dtype=np.float32)

    LS, W0_dist, W1_dist, W2_dist, Beta_tot_dist, area_LS0 = compute_MT_distribs(
        density,
        nbr,
        frac_area,
    )

    return LS, W0_dist, W1_dist, W2_dist, Beta_tot_dist, min_val, max_val, area_LS0


def process_gene(
    gene,
    local_data,
    edges,
    nbr,
    frac_area,
    resolution,
    output_path,
    name,
    n_cov_samples,
    mc_seed,
):
    """
    Process a single gene: build its Minkowski profiles (W0, W1, W2, Beta),
    optionally estimate shot-noise covariance, and save everything to disk.

    Parameters
    ----------
    gene : str
        Gene identifier to process (or subsample label of the form 'X_sub_N').
    local_data : pandas.DataFrame
        Subset of the full transcript table for this MPI rank. Must contain
        columns ['gene', 'global_x', 'global_y'].
    edges : 1D ndarray of float
        Grid edges as returned by ``build_grid``; used for both x and y axes.
    nbr : int
        Number of level sets used to build Minkowski profiles.
    frac_area : float
        Kept for backward compatibility; not used in the current
        normalisation scheme (see ``data_to_Mink``).
    resolution : float
        Grid resolution in the same physical units as 'global_x'/'global_y'.
        Only used to label the output file.
    output_path : str
        Directory where the per-gene .npz file will be written.
    name : str
        Sample name (or subsample name) used in the output filename.
    n_cov_samples : int
        Number of Monte Carlo realisations to draw for shot-noise covariance.
        If <= 0, covariance is not computed.
    mc_seed : int or None
        Optional base seed for Monte Carlo covariance realisations.
        If provided, rerunning with the same inputs reproduces the same
        covariance realisations.

    Notes
    -----
    The output .npz file contains:
      - LS                : 1D array of level-set thresholds,
      - Minkowski_tensor  : array of shape (4, nbr) stacking
                            [W0_dist, W1_dist, W2_dist, Beta_dist],
      - gene_name         : gene identifier used for this run,
      - number_of_transcripts : integer transcript count,
      - min_val, max_val  : pre-normalisation field extrema,
      - area_LS0          : reference excursion-set area at the first level,
      - SN_respl_samples  : optional array of shape (4, n_cov_samples, nbr)
                            with Monte Carlo samples of (W0, W1, W2, Beta).
    """
    fname = f"minkiPy_Minkowski_resolution_{resolution}_{name}_{gene}.npz"
    out_path = io.gene_npz_path(output_path, resolution, name, gene)
    if os.path.exists(out_path):
        print(f"[process_gene] Gene '{gene}' already processed -> Skipping.", flush=True)
        return


    # Extract all transcript positions for this gene on this rank
    filtered_df = local_data[local_data["gene"] == gene]
    x_locations = np.array(filtered_df["global_x"].values)
    y_locations = np.array(filtered_df["global_y"].values)
    n_transcripts = len(x_locations)

    # Compute Minkowski characteristics (profiles over level sets)
    LS, W0_dist, W1_dist, W2_dist, Beta_dist, min_val, max_val, area_LS0 = data_to_Mink(
        x_locations,
        y_locations,
        edges,
        nbr,
        frac_area,
    )
    Minkowski_tensor = np.vstack((W0_dist, W1_dist, W2_dist, Beta_dist))

    # Optionally compute shot-noise resampling covariance
    SN_respl_samples = None
    if n_cov_samples and n_cov_samples > 0:
        SN_respl_samples = SN_respl_covariance(
            x_locations,
            y_locations,
            n_cov_samples,
            edges,
            nbr,
            frac_area,
            mc_seed=mc_seed,
        )

    # For synthetic subsamples, ensure the label encodes the *actual* size:
    # "X_sub_N" where N matches the number of transcripts in this subsample.
    if "_sub_" in gene:
        try:
            target = int(gene.rsplit("_", 1)[-1])
            if target != n_transcripts:
                print(
                    f"[WARN] {gene}: label {target} != count {n_transcripts} -> renomme"
                )
                gene = gene.rsplit("_", 1)[0] + f"_{n_transcripts}"
                out_path = os.path.join(
                    output_path,
                    f"minkiPy_Minkowski_resolution_{resolution}_{name}_{gene}.npz",
                )
        except ValueError:
            # If the suffix is not an integer, we simply leave the gene name as-is
            pass

    # Unified save (works for both original data and subsampled runs)
    save_kwargs = dict(
        LS=LS,
        Minkowski_tensor=Minkowski_tensor,
        gene_name=gene,
        number_of_transcripts=n_transcripts,
        min_val=min_val,
        max_val=max_val,
        area_LS0=area_LS0,
    )
    if SN_respl_samples is not None:
        save_kwargs["SN_respl_samples"] = SN_respl_samples

    io.save_gene_npz(
        output_path,
        resolution,
        name,
        gene,
        LS=LS,
        Minkowski_tensor=Minkowski_tensor,
        number_of_transcripts=n_transcripts,
        min_val=min_val,
        max_val=max_val,
        area_LS0=area_LS0,
        SN_respl_samples=SN_respl_samples,
    )


    # Explicitly drop large arrays to help the GC in long MPI runs
    del (
        filtered_df,
        x_locations,
        y_locations,
        LS,
        W0_dist,
        W1_dist,
        W2_dist,
        Beta_dist,
        Minkowski_tensor,
        SN_respl_samples,
    )


@njit(fastmath=True)
def bilinear_assignment(v00, v10, v01, v11, a, rdm1, rdm2):
    """
    Draw a random position inside a pixel according to a bilinear intensity map.

    The intensities (v00, v10, v01, v11) are defined at the four vertices of
    a square cell of size 'a'. We interpret these as defining a bilinear
    density inside the cell and invert the corresponding CDF using two
    uniform random numbers (rdm1, rdm2).

    Parameters
    ----------
    v00, v10, v01, v11 : float
        Vertex intensities at (0,0), (a,0), (0,a), (a,a) respectively.
    a : float
        Cell size (edge length) in physical units.
    rdm1, rdm2 : float
        Independent uniform random variates in [0, 1).

    Returns
    -------
    x, y : float
        Local coordinates inside the cell in [0, a] × [0, a].
    """
    # Same logic as before, but pass random numbers explicitly to njit
    C1 = v00 + v01
    C2 = -v00 + v10 - v01 + v11
    C3 = v00 + v10 + v01 + v11

    inner_x = C1 * C1 + C2 * C3 * rdm1
    if C2 == 0 or inner_x < 0:
        x = a * 0.5
    else:
        x = a * (-C1 + np.sqrt(inner_x)) / C2

    C1_y = a * (-v00 + v01) + x * (v00 - v01 - v10 + v11)
    C2_y = 2 * a * a * v00 - 2 * a * x * (v00 - v10)
    C3_y = a * a * (a * (v00 + v01) + x * (-v00 - v01 + v10 + v11))

    inner_y = C2_y * C2_y + 4 * C1_y * C3_y * rdm2
    if C1_y == 0 or inner_y < 0:
        y = a * 0.5
    else:
        y = -(C2_y - np.sqrt(inner_y)) / (2 * C1_y)

    return x, y


@njit(boundscheck=False, cache=True, fastmath=True)
def CIC_deposit(x_locations, y_locations, edges):
    """
    Deposit a point cloud onto grid vertices using Cloud-In-Cell (CIC) weights.

    Each point contributes linearly to the four surrounding grid vertices,
    producing a 2D vertex field that later serves as input for expected
    pixel counts.

    Parameters
    ----------
    x_locations, y_locations : 1D arrays of float
        Point coordinates in the same physical units as 'edges'.
    edges : 1D ndarray of float
        Grid edges along one axis; used for both x and y.

    Returns
    -------
    vertex_field : 2D ndarray of float, shape (num_vertices, num_vertices)
        CIC-deposited vertex amplitudes.
    """
    num_vertices = len(edges)
    vertex_field = np.zeros((num_vertices, num_vertices), dtype=np.float32)

    for idx in range(len(x_locations)):
        x, y = x_locations[idx], y_locations[idx]

        # Locate cell index (i, j) such that x is between edges[i] and edges[i+1]
        i = np.searchsorted(edges, x) - 1
        j = np.searchsorted(edges, y) - 1
        i = max(min(i, num_vertices - 2), 0)
        j = max(min(j, num_vertices - 2), 0)

        dx = (x - edges[i]) / (edges[i + 1] - edges[i])
        dy = (y - edges[j]) / (edges[j + 1] - edges[j])

        # Linear weights to four surrounding vertices
        vertex_field[i, j]     += (1 - dx) * (1 - dy)
        vertex_field[i + 1, j] += dx * (1 - dy)
        vertex_field[i, j + 1] += (1 - dx) * dy
        vertex_field[i + 1, j + 1] += dx * dy

    return vertex_field


@njit(boundscheck=False, cache=True, fastmath=True)
def sample_positions(expected_counts, vertex_field, edges, pixel_size, rng_seed):
    """
    Draw a Poisson realisation of a smoothed point process on the grid.

    First, for each pixel we sample a Poisson-distributed count with mean
    given by ``expected_counts``. Then, for each realised count we draw
    positions inside the corresponding cell according to a bilinear density
    derived from the local vertex_field.

    Parameters
    ----------
    expected_counts : 2D ndarray of float, shape (N, N)
        Expected number of points per pixel.
    vertex_field : 2D ndarray of float, shape (N+1, N+1)
        CIC vertex amplitudes; used to define the bilinear density inside
        each pixel.
    edges : 1D ndarray of float
        Grid edges along one axis; used to anchor physical coordinates.
    pixel_size : float
        Side length of a pixel (edges[i+1] - edges[i]) in physical units.
    rng_seed : int
        Seed used to initialise the NumPy random generator (Numba side).

    Returns
    -------
    transcript_positions : 2D ndarray of float, shape (M, 2)
        Coordinates of all sampled points; M is the total Poisson count
        summed over all pixels.
    """
    # ------------------------------------------------------------------
    # Pass-0  ──────────────────────────────────────────────────────────
    # Draw a Poisson count for every pixel and store it while tracking
    # the total number of points to allocate.
    # ------------------------------------------------------------------
    num_pixels = expected_counts.shape[0]
    counts = np.empty((num_pixels, num_pixels), dtype=np.int32)

    np.random.seed(rng_seed)

    total = 0
    for i in range(num_pixels):
        for j in range(num_pixels):
            lam = np.float64(expected_counts[i, j])
            if lam < 0.0:
                lam = 0.0
            c = np.random.poisson(lam)
            counts[i, j] = c
            total += c

    # ------------------------------------------------------------------
    # Pass-1  ──────────────────────────────────────────────────────────
    # Pre-allocate the exact output array and fill it in the same
    # (i, j, within-cell) order as in the original implementation.
    # ------------------------------------------------------------------
    transcript_positions = np.empty((total, 2), dtype=np.float64)
    pos_idx = 0

    for i in range(num_pixels):
        edge_x = edges[i]
        for j in range(num_pixels):
            edge_y = edges[j]
            c = counts[i, j]
            if c == 0:
                continue

            # Vertex values for this pixel
            v00 = vertex_field[i,     j]
            v10 = vertex_field[i + 1, j]
            v01 = vertex_field[i, j + 1]
            v11 = vertex_field[i + 1, j + 1]

            # Draw c positions inside the pixel according to its bilinear density
            for _ in range(c):
                rdm1 = np.random.random()
                rdm2 = np.random.random()
                x_loc, y_loc = bilinear_assignment(
                    v00, v10, v01, v11, pixel_size, rdm1, rdm2
                )
                transcript_positions[pos_idx, 0] = edge_x + x_loc
                transcript_positions[pos_idx, 1] = edge_y + y_loc
                pos_idx += 1

    return transcript_positions


def SN_respl_covariance(
    x_locations,
    y_locations,
    n_cov_samples,
    edges,
    nbr,
    frac_area,
    mc_seed=None,
):
    """
    Estimate shot-noise covariance of Minkowski profiles via Monte Carlo resampling.

    Given an empirical point pattern (x_locations, y_locations), this routine:
      1) Deposits the points onto grid vertices using CIC.
      2) Builds a pixel-wise expected count map.
      3) Draws ``n_cov_samples`` independent Poisson realisations of the field.
      4) For each realisation, recomputes Minkowski profiles using
         ``data_to_Mink``.

    The resulting array ``overall_samples`` collects the Monte Carlo
    fluctuations of (W0, W1, W2, Beta) across level sets, which can be used
    to estimate covariance matrices or confidence bands.

    Parameters
    ----------
    x_locations, y_locations : 1D arrays of float
        Coordinates of the original point pattern.
    n_cov_samples : int
        Number of Monte Carlo resamples to generate.
    edges : 1D ndarray of float
        Grid edges along one axis, used for both x and y.
    nbr : int
        Number of level sets in the Minkowski profiles.
    frac_area : float
        Kept for backward compatibility; not used in the current
        normalisation scheme.
    mc_seed : int or None, optional
        Optional base seed used to make Monte Carlo realisations reproducible
        across runs.

    Returns
    -------
    overall_samples : ndarray of float, shape (4, n_cov_samples, nbr)
        Monte Carlo samples of Minkowski profiles:
        overall_samples[0, r, :] = W0 for realisation r
        overall_samples[1, r, :] = W1 for realisation r
        overall_samples[2, r, :] = W2 for realisation r
        overall_samples[3, r, :] = Beta for realisation r
    """
    total_positions = len(x_locations)
    num_vertices = len(edges)
    num_pixels = num_vertices - 1
    pixel_size = edges[1] - edges[0]

    # CIC deposition of the empirical point pattern onto vertices
    vertex_field = CIC_deposit(x_locations, y_locations, edges)

    # Expected pixel counts from vertex values (sum of four neighbouring vertices)
    expected_counts = (
        vertex_field[:-1, :-1]
        + vertex_field[1:, :-1]
        + vertex_field[:-1, 1:]
        + vertex_field[1:, 1:]
    )
    expected_counts *= total_positions / np.sum(expected_counts)

    overall_samples = np.empty((4, n_cov_samples, nbr), dtype=np.float32)
    if mc_seed is None:
        # No explicit seed from user -> non-deterministic run-level entropy.
        ss = np.random.SeedSequence()
    else:
        # Explicit user seed -> deterministic reproducibility across runs.
        ss = np.random.SeedSequence(int(mc_seed) & 0xFFFFFFFF)
    seeds = ss.generate_state(n_cov_samples, dtype=np.uint32)

    for r in range(n_cov_samples):
        # Draw a Poisson realisation of the underlying smoothed process
        transcript_positions = sample_positions(
            expected_counts,
            vertex_field,
            edges,
            pixel_size,
            rng_seed=int(seeds[r]),
        )

        if len(transcript_positions) == 0:
            particle_x_coords = np.array([])
            particle_y_coords = np.array([])
        else:
            particle_x_coords = transcript_positions[:, 0]
            particle_y_coords = transcript_positions[:, 1]

        # Compute Minkowski profiles for this realisation
        _, W0, W1, W2, Beta, _, _, _ = data_to_Mink(
            particle_x_coords,
            particle_y_coords,
            edges,
            nbr,
            frac_area,
        )
        overall_samples[0, r, :] = W0
        overall_samples[1, r, :] = W1
        overall_samples[2, r, :] = W2
        overall_samples[3, r, :] = Beta

    return overall_samples


    



def grid_definition(resolution, processed_df=None, processed_adata=None):
    """
    Define a square grid that covers all transcript coordinates.

    Parameters
    ----------
    resolution : float
        Grid cell size in the same physical units as 'global_x'/'global_y'.
    processed_df : pandas.DataFrame, optional
        DataFrame with columns ['global_x', 'global_y'] giving transcript
        positions. This is the only argument used at the moment.
    processed_adata : AnnData, optional
        Reserved for potential future use; ignored in the current implementation.

    Returns
    -------
    xmin : float
        Leftmost coordinate of the grid in x.
    grid_size : int
        Number of pixels along one axis (grid is square: grid_size × grid_size).
    edges : 1D ndarray of float
        Array of length grid_size + 1 with cell edges, used for both x and y.
    """
    xmin = np.array(processed_df["global_x"].values).min()
    xmax = np.array(processed_df["global_x"].values).max()
    ymin = np.array(processed_df["global_y"].values).min()
    ymax = np.array(processed_df["global_y"].values).max()

    # We enforce a square domain spanning [xmin, xmin + grid_size * resolution]
    spread_max = max(xmax, ymax)
    grid_size = int(np.ceil(spread_max / resolution))
    edges = np.linspace(xmin, xmin + grid_size * resolution, grid_size + 1)
    return xmin, grid_size, edges



def compute_area_mask_from_transcripts(transcripts_df, edges):
    """
    Estimate the area of the tissue mask from transcript positions.

    We place the square grid defined by ``edges`` over the transcript cloud,
    mark all cells that contain at least one transcript, and return the
    total area covered by these occupied cells.

    Parameters
    ----------
    transcripts_df : pandas.DataFrame
        Must contain columns ['global_x', 'global_y'] for each transcript.
    edges : 1D ndarray of float
        Grid edges along one axis, used for both x and y.

    Returns
    -------
    area_mask : float
        Occupied area, in the same units as edges², computed as
        (number of occupied pixels) × (cell_area).
    """
    xs = transcripts_df["global_x"].values
    ys = transcripts_df["global_y"].values

    # reuse the exact edges from grid_definition:
    x_edges = edges
    y_edges = edges

    # bin each point into a cell index
    ix = np.searchsorted(x_edges, xs, side="right") - 1
    iy = np.searchsorted(y_edges, ys, side="right") - 1

    # keep only valid cell indices: 0 .. len(edges)-2
    max_cell = len(edges) - 2
    ix = np.clip(ix, 0, max_cell)
    iy = np.clip(iy, 0, max_cell)

    # unique occupied cells
    unique_cells = set(zip(ix, iy))
    n_pixels = len(unique_cells)

    # each cell has area (edges[1] - edges[0])²
    cell_area = (edges[1] - edges[0]) ** 2
    return n_pixels * cell_area



#___________________________________________________________________________________________________________________________________________________________________________________________________

# Driver (ne s'exécute que si on lance le fichier comme un script)
#___________________________________________________________________________________________________________________________________________________________________________________________________


# ---------------------------------------------------------------------
# Re-export MPI orchestration helpers from the dedicated module.
# This avoids code duplication while keeping backward compatibility
# for the historical driver in minkowski_core.main().
# ---------------------------------------------------------------------





if __name__ == "__main__":
    main()
