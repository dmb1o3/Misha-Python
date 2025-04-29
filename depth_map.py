import cv2
import argparse
import itertools
import numpy as np
import tkinter as tk
from tkinter import filedialog
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import normalize

import plotly.graph_objects as go

"""
Taken from

https://github.com/gray0018/Discrete-normal-integration
"""

eps = np.finfo(float).eps  # epsilon for avoiding zero division

# command line parser
parser = argparse.ArgumentParser(description='Normal Integration by solving Poisson Equation')
parser.add_argument('normal', help='the path of normal map')
parser.add_argument('-d', '--depth', default=None, help='the path of depth prior')
parser.add_argument('--d_lambda', type=float, default=100, help='how much will the depth prior influence the result')
parser.add_argument('-o', '--output', default='output', help='name of the output object and depth map')
parser.add_argument('--obj', dest='write_obj', action='store_const',
                    const=True, default=False, help='write wavefront obj file, by default False')


def write_obj(filename, d, d_ind):
    obj = open(filename, "w")
    h, w = d.shape

    x = np.arange(0.5, w, 1)
    y = np.arange(h - 0.5, 0, -1)
    xx, yy = np.meshgrid(x, y)
    mask = d_ind > 0

    xyz = np.vstack((xx[mask], yy[mask], d[mask])).T
    obj.write(''.join(["v {0} {1} {2}\n".format(x, y, z) for x, y, z in xyz]))  # write vertices into obj file

    right = np.roll(d_ind, -1, axis=1)
    right[:, -1] = 0
    right_mask = right > 0

    down = np.roll(d_ind, -1, axis=0)
    down[-1, :] = 0
    down_mask = down > 0

    rd = np.roll(d_ind, -1, axis=1)
    rd = np.roll(rd, -1, axis=0)
    rd[-1, :] = 0
    rd[:, -1] = 0
    rd_mask = rd > 0

    up_tri = mask & rd_mask & right_mask  # counter clockwise
    low_tri = mask & down_mask & rd_mask  # counter clockwise

    xyz = np.vstack((d_ind[up_tri], rd[up_tri], right[up_tri])).T
    obj.write(
        ''.join(["f {0} {1} {2}\n".format(x, y, z) for x, y, z in xyz]))  # write upper triangle facet into obj file
    xyz = np.vstack((d_ind[low_tri], down[low_tri], rd[low_tri])).T
    obj.write(
        ''.join(["f {0} {1} {2}\n".format(x, y, z) for x, y, z in xyz]))  # write lower triangle facet into obj file

    obj.close()


class PoissonOperator(object):

    def __init__(self, data, mask, depth_info=None, depth_weight=0.1):
        h, w = mask.shape

        self.index_1d = np.ones([h, w]) * (-1)

        self.data = data
        self.mask = mask
        self.window_shape = (3, 3)
        self.valid_index = np.where(self.mask.ravel() != 0)[0]
        self.valid_num = len(self.valid_index)
        self.index_1d.reshape(-1)[self.valid_index] = np.arange(self.valid_num)

        self.v_count = (self.mask.astype(np.int32)).sum()  # total number of all vertices
        self.v_index = np.zeros_like(self.mask, dtype='uint')  # indices for all vertices
        self.v_index[self.mask.astype(np.bool_)] = np.arange(self.v_count) + 1

        self.depth = np.zeros([h, w])

        self.f_4neighbor = lambda x: np.array([x[1, 1], x[1, 0], x[2, 1], x[0, 1], x[1, 2]])

        # add depth_info and depth_weight for depth fusion
        self.depth_A = None
        self.depth_b = None
        if depth_info is not None:
            self.depth_A, self.depth_b = self.add_depth_info(depth_info, depth_weight)

    def add_depth_info(self, depth, w):
        rows, cols = depth.shape
        r = 0
        ind = 0
        col = []
        b = []
        variable_num = int(np.sum(self.mask))
        for i in range(rows):
            for j in range(cols):
                if self.mask[i, j]:
                    ind += 1
                if ~np.isnan(depth[i, j]):
                    r += 1
                    col.append(ind)
                    b.append(w * depth[i][j])

        data = np.array([w for i in range(r)])
        row = np.array([i for i in range(r)])
        col = np.array(col)

        A = sparse.coo_matrix((data, (row, col)), shape=(r, variable_num))
        b = np.array(b)
        return A, b

    def build_patch_for_poisson(self, mask_patch, data_patch, position_patch, weight=1):
        """
        get the cols and val for sparse matrix in this single patch
        :param mask_patch: 3*3 with weight
        :param data_patch: 3*3*d d is the dimension of the data, in normal case, we only need to input [p, q] 2d data
        :param position_patch: 3*3*1 the 1D patch position in the global image coordinate in 1d
        :param weight: the weight for this rows, which determine how important of this row
        :return: [colidx, colvals, bvals] colidx and colvals in 1d array with the same length, bval is a scaler
        """

        mask_used = self.f_4neighbor(mask_patch)
        data_used = self.f_4neighbor(data_patch)
        position_used = self.f_4neighbor(position_patch)

        colidx = []
        colvals = []
        bvals = 0

        if mask_used[1] == 1:
            D_ct = - (data_used[0] + data_used[1])[0] / 2  # the val between center to top
            colidx.append(position_used[1])
            colvals.append(1)
            bvals += D_ct
        if mask_used[2] == 1:
            D_cl = - (data_used[0] + data_used[2])[1] / 2  # the val between center to left
            colidx.append(position_used[2])
            colvals.append(1)
            bvals += D_cl
        if mask_used[3] == 1:
            D_cr = (data_used[0] + data_used[3])[1] / 2  # the val between center to right
            colidx.append(position_used[3])
            colvals.append(1)
            bvals += D_cr
        if mask_used[4] == 1:
            D_cb = (data_used[0] + data_used[4])[0] / 2  # the val between center to bottom
            colidx.append(position_used[4])
            colvals.append(1)
            bvals += D_cb

        colidx.append(position_used[0])
        colvals.append(- np.sum(np.array(colvals)))

        return [colidx, colvals, bvals]

    def get_patches(self):
        # step 1: padding the data
        mask_pad = cv2.copyMakeBorder(self.mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        from sklearn.feature_extraction.image import extract_patches_2d
        self.mask_patches = extract_patches_2d(mask_pad, self.window_shape)

        index_1d_pad = cv2.copyMakeBorder(self.index_1d, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=-1)
        self.index_1d_patches = extract_patches_2d(index_1d_pad, self.window_shape)
        data_pad = cv2.copyMakeBorder(self.data, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        self.data_patches = extract_patches_2d(data_pad, self.window_shape)

    def run(self):
        self.get_patches()
        self.poisson_b = []
        cols_all = []
        vals_all = []
        rows_all = []

        row_global = 0
        for i in self.valid_index:
            [colidx, colvals, bvals] = self.build_patch_for_poisson(self.mask_patches[i], self.data_patches[i],
                                                                    self.index_1d_patches[i])
            self.poisson_b.append(bvals)
            cols_all.append(colidx)
            vals_all.append(colvals)
            rows_all.append(np.ones_like(colidx) * row_global)
            row_global += 1
        rows_all_flat = list(itertools.chain.from_iterable(rows_all))
        cols_all_flat = list(itertools.chain.from_iterable(cols_all))
        vals_all_flat = list(itertools.chain.from_iterable(vals_all))

        self.poisson_A = sparse.coo_matrix((vals_all_flat, (rows_all_flat, cols_all_flat)),
                                           shape=(row_global, self.valid_num))
        self.poisson_b = np.array(self.poisson_b)

        # depth fusion
        if self.depth_A is not None:
            self.poisson_A = sparse.vstack((self.poisson_A, self.depth_A))
            self.poisson_b = np.hstack((self.poisson_b, self.depth_b))

        depth = spsolve(self.poisson_A.T @ self.poisson_A, self.poisson_A.T @ self.poisson_b)
        self.depth.reshape(-1)[self.valid_index] = depth
        return self.depth


def read_normal_map(path):
    '''
    description:
        read a normal map(jpg, png, bmp, etc.), and convert it to an normalized (x,y,z) form
    input:
        path: path of the normal map
    output:
        n: normalized normal map
        mask_bg: background mask
    '''

    if ".npy" in path:
        n = np.load(path)
        mask_bg = (n[..., 2] == 0)  # get background mask
    else:
        n = cv2.imread(path)

        n[..., 0], n[..., 2] = n[..., 2], n[..., 0].copy()  # Change BGR to RGB
        #mask_bg = (n[..., 2] == 0)  # get background mask
        mask_bg = (n[..., 2] == 128)  # get background mask
        n = n.astype(np.float32)  # uint8 -> float32

        # x,y:[0,255]->[-1,1] z:[128,255]->[0,1]
        n[..., 0] = n[..., 0] * 2 / 255 - 1
        n[..., 1] = n[..., 1] * 2 / 255 - 1
        n[..., 2] = (n[..., 2] - 128) / 127

        n = normalize(n.reshape(-1, 3)).reshape(n.shape)

    # fill background with [0,0,0]
    n[mask_bg] = [0, 0, 0]
    return n, ~mask_bg


def write_depth_map(filename, depth, mask_bg):
    depth[mask_bg] = np.nan
    np.save(filename, depth)


def remove_polynomial_trend(depth_map, degree=2):
    """
    Given a depth map and degrees will fit a polynomial to the depth map to try and remove noise from image

    :param depth_map: Numpy array of depth map
    :param degree: Degrees of polynomial we are fitting
    :return: Returns the corrected depth map and the fitted polynomial
    """
    # Replace 0 with NaN
    depth_map = np.where(depth_map == 0, np.nan, depth_map)
    # Create a mask for valid (non-NaN) points
    valid_mask = ~np.isnan(depth_map)

    # Extract only the valid coordinates and depth values
    y_indices, x_indices = np.where(valid_mask)
    valid_Z = depth_map[valid_mask]

    # Create a new compact depth map that only contains valid points
    # First, find the bounding box of valid points
    min_y, max_y = y_indices.min(), y_indices.max()
    min_x, max_x = x_indices.min(), x_indices.max()

    # Calculate dimensions of the compact depth map
    compact_height = max_y - min_y + 1
    compact_width = max_x - min_x + 1

    # Create the compact depth map filled with NaN
    depth_map = np.full((compact_height, compact_width), np.nan)

    # Fill in the valid values (shifted to start at 0,0)
    depth_map[y_indices - min_y, x_indices - min_x] = valid_Z
    height, width = depth_map.shape

    # Create coordinate grids
    Y_grid, X_grid = np.mgrid[:height, :width]
    x = X_grid.flatten()
    y = Y_grid.flatten()
    z = depth_map.flatten()

    # Ignore NaN values
    valid_indices = ~np.isnan(z)
    x = x[valid_indices]
    y = y[valid_indices]
    z = z[valid_indices]

    # Normalize coordinates
    x_mean, x_std = x.mean(), x.std()
    y_mean, y_std = y.mean(), y.std()
    x_norm = (x - x_mean) / x_std
    y_norm = (y - y_mean) / y_std
    z_mean = z.mean()
    z_std = z.std()
    z_norm = (z - z_mean) / z_std

    # Generate polynomial terms
    powers = [(i, j) for i in range(degree + 1) for j in range(degree + 1 - i)]
    A = np.column_stack([x_norm ** i * y_norm ** j for i, j in powers])

    # Fit polynomial using lstsq (faster)
    coeffs, _, _, _ = np.linalg.lstsq(A, z_norm, rcond=None)

    # Define polynomial function for reconstruction
    def poly_func(x, y, coeffs):
        return sum(
            coeffs[k] * ((x - x_mean) / x_std) ** i * ((y - y_mean) / y_std) ** j for k, (i, j) in enumerate(powers))

    # Reconstruct fitted surface
    fitted_surface = poly_func(X_grid, Y_grid, coeffs) * z_std + z_mean

    # Subtract fitted surface
    corrected_depth_map = depth_map - fitted_surface

    return depth_map, corrected_depth_map, fitted_surface


def generate_depth_map(normal_path, output, degree=2, depth=None, d_lambda=100):
    print("Start reading input data...")
    n, mask = read_normal_map(normal_path)
    if depth is not None:
        depth = np.load(depth)

    p = -n[..., 0] / (n[..., 2] + eps)  # avoid zero devision
    q = -n[..., 1] / (n[..., 2] + eps)  # avoid zero devision

    task = PoissonOperator(np.dstack([p, q]), mask.astype(np.int8), depth, d_lambda)
    print("Start normal integration...")
    d = task.run()
    d = np.where(d == 0, np.nan, d)
    mask = ~np.isnan(d)
    d[mask] = d[mask] - d[mask].min()
    # Shift by the minimum to ensure that the z values do not go below 0
    print("Start polynomial fitting...")
    d, c_d, fitted = remove_polynomial_trend(d, degree=degree)
    mask = ~np.isnan(d)
    c_mask = ~np.isnan(c_d)

    print("Start writing depth map...")
    write_depth_map("{0}.npy".format(output), d, ~mask)  # write depth file
    write_depth_map("{0}_corrected.npy".format(output), c_d, ~c_mask)  # write depth file
    # Setup for blender object
    v_count = (mask.astype(np.int32)).sum()
    v_index = np.zeros_like(mask, dtype='uint')  # indices for all vertices
    v_index[mask.astype(np.bool_)] = np.arange(v_count) + 1
    # Write blender object
    write_obj("{0}.obj".format(output), d, v_index) # Write blender object file
    # Setup for blender object
    v_count = (c_mask.astype(np.int32)).sum()
    v_index = np.zeros_like(c_mask, dtype='uint')  # indices for all vertices
    # Write blender object
    v_index[mask.astype(np.bool_)] = np.arange(v_count) + 1
    write_obj("{0}_corrected.obj".format(output), c_d, v_index) # Write corrected blender object file
    print("Finish!")

    return d, c_d, fitted


if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()
    # Prompt user to set directory with images
    directory = filedialog.askdirectory() + '/'
    normal_path = directory + "normals/normal_map.npy"
    output = directory + "depth_map"

    d_lambda = 100
    degree = input("How many degrees would you like to fit polynomial to: ")
    degree = int(degree)
    print("Start reading input data...")
    n, mask = read_normal_map(normal_path)

    p = -n[..., 0] / (n[..., 2] + eps)  # avoid zero devision
    q = -n[..., 1] / (n[..., 2] + eps)  # avoid zero devision

    task = PoissonOperator(np.dstack([p, q]), mask.astype(np.int8), None, d_lambda)
    print("Start normal integration...")
    d = task.run()
    d = np.where(d == 0, np.nan, d)
    mask = ~np.isnan(d)
    d[mask] = d[mask] - d[mask].min()
    # Shift by the minimum to ensure that the z values do not go below 0
    print("Start polynomial fitting...")
    d, c_d, fitted = remove_polynomial_trend(d, degree=degree)
    mask = ~np.isnan(d)
    c_mask = ~np.isnan(c_d)

    print("Start writing depth map...")
    write_depth_map("{0}.npy".format(output), d, ~mask)  # write depth file
    write_depth_map("{0}_corrected.npy".format(output), c_d, ~c_mask)  # write corrected depth file
    # Setup for blender object
    v_count = (mask.astype(np.int32)).sum()
    v_index = np.zeros_like(mask, dtype='uint')  # indices for all vertices
    v_index[mask.astype(np.bool_)] = np.arange(v_count) + 1
    write_obj("{0}.obj".format(output), d, v_index) # Write blender object file
    # Setup for blender object
    v_count = (c_mask.astype(np.int32)).sum()
    v_index = np.zeros_like(c_mask, dtype='uint')  # indices for all vertices
    v_index[mask.astype(np.bool_)] = np.arange(v_count) + 1
    write_obj("{0}_corrected.obj".format(output), c_d, v_index) # Write corrected blender object file
    print("Finish!")

    # Create figure
    fig = go.Figure()
    # Add original depth map
    fig.add_trace(go.Surface(z=d, colorscale='gray', opacity=0.7, name="Original Depth"))
    # Add fitted polynomial surface
    fig.add_trace(go.Surface(z=fitted, colorscale='viridis', opacity=0.6, name="Fitted Curve"))
    # Update layout
    fig.update_layout(title='Original Depth Map vs Fitted Polynomial Surface', autosize=True)
    fig.show()
    # Display the corrected depth map
    fig2 = go.Figure()
    fig2 = go.Figure(data=[go.Surface(z=c_d, colorscale='gray')])
    # fig2.update_layout(title='Corrected Depth Map', autosize=True)
    fig2.update_layout(title='Corrected Depth Map',
                       scene=dict(zaxis=dict(range=[c_d.min(), 100])), autosize=True)
    fig2.show()
    # Display Original depth map
    fig3 = go.Figure()
    fig3 = go.Figure(data=[go.Surface(z=d, colorscale='gray')])
    fig3.update_layout(title='Depth Map', autosize=True)
    fig3.show()
