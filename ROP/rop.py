import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.integrate import quad
import os
import glob
import json
import argparse

import pandas as pd

from skimage.io      import imread
from skimage.draw    import polygon
from skimage.morphology import skeletonize

import networkx as nx
import itertools

def arc_length(px, py, *args):
    """
    Compute the total arc length and per-segment lengths of a curve.

    Usage:
        arclen, seglen = arc_length(px, py)
        arclen, seglen = arc_length(px, py, pz, ...)            # 3D or higher-D
        arclen, seglen = arc_length(px, py, 'pchip')            # choose method
        arclen, seglen = arc_length(px, py, pz, 'spline')       # mix coords & method

    Parameters
    ----------
    px, py : array-like, shape (n,)
        Coordinates of the first two dimensions.
    *args : array-like or str, optional
        - Additional coordinate arrays (must each be length n).
        - Or one string among {'linear','pchip','spline'} to select method.
          Default is 'linear'.

    Returns
    -------
    arclen : float
        Total arc length.
    seglen : ndarray, shape (n-1,)
        Arc length of each chord segment.

    """
    # --- validate px, py ---
    px = np.asarray(px)
    py = np.asarray(py)
    if px.ndim != 1 or py.ndim != 1 or px.shape != py.shape:
        raise ValueError("px and py must be 1D arrays of equal length")
    n = px.size
    if n < 2:
        raise ValueError("Need at least two points to compute arc length")

    # --- parse varargin equivalents ---
    method = 'linear'
    coords = [px, py]
    for a in args:
        if isinstance(a, str):
            method = a.lower()
            if method not in ('linear', 'pchip', 'spline'):
                raise ValueError("Invalid method: choose 'linear','pchip', or 'spline'")
        else:
            arr = np.asarray(a)
            if arr.shape != px.shape:
                raise ValueError("All coordinate arrays must match length of px,py")
            coords.append(arr)

    # stack into an (n × d) array
    data = np.vstack(coords).T   # shape = (n, nd)
    nd = data.shape[1]

    # --- linear chord lengths ---
    diffs = np.diff(data, axis=0)           # shape = (n-1, nd)
    seglen = np.linalg.norm(diffs, axis=1)  # euclidean per segment
    arclen = seglen.sum()
    if method == 'linear':
        return arclen, seglen

    # --- build parameterization by cumulative chord length ---
    t = np.empty(n)
    t[0] = 0.0
    t[1:] = np.cumsum(seglen)

    # --- build interpolators on t for each dim ---
    if method == 'pchip':
        interps = [PchipInterpolator(t, data[:, i]) for i in range(nd)]
    else:  # 'spline'
        # CubicSpline defaults to “not-a-knot” end conditions (same as MATLAB)
        interps = [CubicSpline(t, data[:, i], bc_type='not-a-knot')
                   for i in range(nd)]

    # --- integrate speed = sqrt(sum((dx_i/dt)^2)) over each segment ---
    seglen_int = np.zeros_like(seglen)
    def speed(tt):
        # vectorized speed at times tt
        ssq = np.zeros_like(tt, dtype=float)
        for interp in interps:
            dval = interp.derivative()(tt)
            ssq += dval**2
        return np.sqrt(ssq)

    for i in range(n-1):
        seglen_int[i], _ = quad(speed, t[i], t[i+1])

    return seglen_int.sum(), seglen_int


def mean_distance_measure(x, y, isshow=False):
    """
    Compute the local distance measure (DM) between inflection points of a 2D curve,
    and return its mean plus the number of inflection points (ipf).

    Parameters
    ----------
    x, y : array-like, shape (n,)
        Coordinates of the vessel centerline (n >= 2).
    isshow : bool, optional
        If True, overlay the detected inflection segments on a plot.

    Returns
    -------
    mean_dm : float
        The average local DM between inflection points (or 1 if no inflections).
    ipf : int
        The number of inflection points detected.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("x and y must be 1D arrays of the same length")
    n = x.size
    if n < 2:
        raise ValueError("Need at least two points")

    # 1st derivatives
    dx = np.diff(x)
    dy = np.diff(y)
    # central‐difference for 2nd derivative
    dx2 = (dx[:-1] + dx[1:]) / 2
    dy2 = (dy[:-1] + dy[1:]) / 2
    # truncate dx, dy to match dx2/dy2 length
    dx = dx[:-1]
    dy = dy[:-1]

    # curvature k at each interior point
    denom = (dx**2 + dy**2)**1.5
    k = ((dx * dy2) - (dx2 * dy)) / (denom + np.finfo(float).eps)

    N = 0              # count of inflection points
    DM = []            # list of local distance measures
    previous_pt = None

    # detect sign changes in curvature
    for i in range(len(k) - 1):
        if k[i] * k[i+1] < 0:
            N += 1
            idx = i + 1  # Python zero‐based index of the inflection pt
            if N == 1:
                # first segment: start → first inflection
                chord = np.hypot(x[idx] - x[0], y[idx] - y[0])
                arc, _ = arc_length(x[:idx+1], y[:idx+1])
                DM.append(arc / chord)
                previous_pt = idx

                if isshow:
                    plt.plot(x[0], y[0], 'or', linewidth=2)
                    plt.plot(x[idx], y[idx], 'or', linewidth=2)
                    plt.plot([x[0], x[idx]], [y[0], y[idx]], linewidth=2)
                    plt.plot(x[:idx+1], y[:idx+1], 'r-')
                    plt.pause(0.05)
            else:
                # intermediate segment: inflection → next inflection
                chord = np.hypot(x[idx] - x[previous_pt], y[idx] - y[previous_pt])
                arc, _ = arc_length(x[previous_pt:idx+1], y[previous_pt:idx+1])
                DM.append(arc / chord)

                if isshow:
                    plt.plot(x[previous_pt], y[previous_pt], 'or', linewidth=2)
                    plt.plot(x[idx], y[idx], 'or', linewidth=2)
                    plt.plot([x[previous_pt], x[idx]],
                             [y[previous_pt], y[idx]], linewidth=2)
                    plt.plot(x[previous_pt:idx+1],
                             y[previous_pt:idx+1], 'r-')
                    plt.pause(0.2)

                previous_pt = idx

    # last segment: last inflection → end
    if N >= 1:
        chord = np.hypot(x[-1] - x[previous_pt], y[-1] - y[previous_pt])
        arc, _ = arc_length(x[previous_pt:], y[previous_pt:])
        DM.append(arc / chord)

        if isshow:
            plt.plot(x[previous_pt], y[previous_pt], 'or', linewidth=2)
            plt.plot(x[-1], y[-1], 'or', linewidth=2)
            plt.plot([x[previous_pt], x[-1]],
                     [y[previous_pt], y[-1]], linewidth=2)
            plt.plot(x[previous_pt:], y[previous_pt:], 'r-')

        # mirror MATLAB’s removal of the global segments
        # so we only average the local inflection-to-inflection DMs
        DM = DM[1:-1]
        ipf = len(DM) + 1  # yields the original N
    else:
        # no inflection: single global DM
        chord = np.hypot(x[-1] - x[0], y[-1] - y[0])
        arc, _ = arc_length(x, y)
        DM = [arc / chord]
        ipf = 1

    mean_dm = float(np.nanmean(DM))
    if np.isnan(mean_dm):
        mean_dm = 1.0

    return mean_dm, ipf


def num_critical_pts(x, y, isshow=False):
    x = np.asarray(x); y = np.asarray(y)
    dx = np.diff(x)
    dy = np.diff(y)
    # angle of each segment (radians)
    angles = np.arctan2(dy, dx)
    # slope = tan(angle), but we really only need sign(angles)
    sign = np.sign(angles)

    # detect sign changes in the tangent angle
    changes = sign[:-1] * sign[1:] < 0
    N = int(np.count_nonzero(changes))
    if isshow and N > 0:
        # plot the points where the sign flip occurred
        idxs = np.nonzero(changes)[0]
        for ii in idxs:
            plt.plot(x[ii], y[ii], 'og', markersize=12, linewidth=4)
            plt.pause(0.1)

    # ensure at least 1
    return N if N > 0 else 1


def sd_theta(x, y, isshow=False):
    x = np.asarray(x); y = np.asarray(y)
    dx = np.diff(x)
    dy = np.diff(y)
    # angle in degrees
    theta = np.degrees(np.arctan2(dy, dx))

    if isshow:
        # if you really want to plot each tangent line:
        for k in range(len(theta)):
            # just draw the line through (x[k],y[k]) at angle theta[k]
            length = max(np.ptp(x), np.ptp(y))
            dxl = length * np.cos(np.radians(theta[k]))
            dyl = length * np.sin(np.radians(theta[k]))
            plt.plot([x[k] - dxl, x[k] + dxl],
                     [y[k] - dyl, y[k] + dyl],
                     ':k', linewidth=0.5)
            plt.pause(1e-19)

    SD = np.std(np.abs(theta)) / 100.0
    return SD, theta

def vessel_tortuosity_index(x, y, isshow=False):
    """
    Compute the Vessel Tortuosity Index (VTI) and intermediate measures for a 2D curve.

    Parameters
    ----------
    x, y : array-like, shape (n,)
        Coordinates of the curve centerline (n >= 2).
    isshow : bool, optional
        If True, display graphical overlays of the intermediate steps.

    Returns
    -------
    VTI : float
        Vessel Tortuosity Index.
    sd : float
        Standard deviation of tangent angles (scaled by 1/100).
    mean_dm : float
        Mean local distance measure between inflection points.
    num_inflection_pts : int
        Number of inflection points.
    num_cpts : int
        Number of critical points.
    len_arch : float
        Arc length of the full curve.
    len_cord : float
        Straight‐line chord length between endpoints.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("x and y must be 1D arrays of the same length")
    if x.size < 2:
        raise ValueError("Need at least two points")

    # Optional plotting of the raw curve
    if isshow:
        plt.figure()
        plt.plot(x, y, 'k', linewidth=2)
        ax = plt.gca()
        # mimic MATLAB's "box off"
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.axis([x.min(), x.max(), y.min() - 3, y.max() + 3])
        plt.title("Vessel Centerline with Inflections/Criticals")

    # chord length (endpoint‐to‐endpoint)
    len_cord = np.hypot(x[-1] - x[0], y[-1] - y[0])

    # arc length of the entire curve
    len_arch, _ = arc_length(x, y)

    # SD of tangent‐angle deviations
    sd, _ = sd_theta(x, y, isshow)

    # mean distance measure and count of inflection points
    mean_dm, num_inflection_pts = mean_distance_measure(x, y, isshow)

    # number of critical points
    num_cpts = num_critical_pts(x, y, isshow)

    # final Vessel Tortuosity Index
    VTI = (len_arch * sd * num_cpts * mean_dm) / len_cord

    return VTI, sd, mean_dm, num_inflection_pts, num_cpts, len_arch, len_cord

def skeleton_to_graph(skel):
    G = nx.Graph()
    ys, xs = np.nonzero(skel)
    nodes = list(zip(ys, xs))
    G.add_nodes_from(nodes)
    for y, x in nodes:
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dy==0 and dx==0: continue
                ny, nx2 = y+dy, x+dx
                if 0 <= ny < skel.shape[0] and 0 <= nx2 < skel.shape[1] and skel[ny, nx2]:
                    G.add_edge((y,x), (ny,nx2))
    return G

def image_to_centerline_coords(img_path):
    # 1) load & segment
    img = imread(img_path)
    gray = rgb2gray(img)
    thresh = threshold_otsu(gray)
    mask = gray > thresh
    mask = closing(mask, disk(2))          # optional cleanup

    # 2) skeletonize
    skel = skeletonize(mask)

    # 3) build an 8‑neighbor graph
    G = nx.Graph()
    ys, xs = np.nonzero(skel)
    for y, x in zip(ys, xs):
        G.add_node((y, x))
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == dx == 0: continue
                ny, nx2 = y + dy, x + dx
                if (0 <= ny < skel.shape[0]
                and 0 <= nx2 < skel.shape[1]
                and skel[ny, nx2]):
                    G.add_edge((y, x), (ny, nx2))

    # 4) pick the largest component’s longest path
    #    (you can iterate components if there are many vessels)
    comp = max(nx.connected_components(G), key=len)
    subG = G.subgraph(comp)
    ends = [n for n,d in subG.degree() if d == 1]
    if len(ends) < 2:
        raise RuntimeError("no endpoints found")
    best_path = []
    max_len = 0
    for a, b in itertools.combinations(ends, 2):
        path = nx.shortest_path(subG, a, b)
        if len(path) > max_len:
            max_len, best_path = len(path), path

    # 5) unzip into x, y arrays (flip because our nodes are (row=y, col=x))
    ys, xs = zip(*best_path)
        # compute VTI
        VTI, *_ = vessel_tortuosity_index(xs, ys, isshow=False)
        vtis.append(VTI)

    return vtis

def main(directory, out_csv, ext="png"):
    records = []
    # assume JSONs are named foo.json, images foo.jpg
    for image_path in sorted(glob.glob(os.path.join(directory, "*.png"))):
        if not os.path.exists(image_path):
            print(f"Warning: no image for {image_path}, skipping")
            continue

        vtis = image_to_centerline_coords(image_path)
        mean_vti = float(np.nan) if not vtis else float(np.mean(vtis))
        records.append({
            "image": os.path.basename(jpg_path),
            "mean_VTI": mean_vti,
            "all_VTIs": vtis
        })

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"Processed {len(records)} images → {out_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Batch VTI from PNGs"
    )
    p.add_argument("--dir",     required=True,
                   help="Directory with image paths")
    p.add_argument("--out",     required=True,
                   help="Output CSV file")
    p.add_argument("--ext",     default="png",
                   help="Image extension (default: png)")
    args = p.parse_args()

    main(args.dir, args.out, ext=args.ext)
    
    

