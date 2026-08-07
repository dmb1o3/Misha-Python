import cv2
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


def write_obj(filename, d, d_ind, obj_scale=1.0):
    print(f"write_obj called for {filename}")
    print(f"d shape: {d.shape}, dtype: {d.dtype}")
    print(f"d_ind shape: {d_ind.shape}, dtype: {d_ind.dtype}")
    print(f"d range: {np.nanmin(d):.3f} to {np.nanmax(d):.3f}")
    print(f"NaN count: {np.isnan(d).sum()}, Inf count: {np.isinf(d).sum()}")
    print(f"d_ind sum (masked pixels): {d_ind.sum()}")

    # Check if d_ind is boolean or int
    if d_ind.dtype != bool:
        print(f"WARNING: d_ind is {d_ind.dtype}, converting to bool")
        d_ind = d_ind > 0
    obj = open(filename, "w")
    h, w = d.shape

    # Pixel size in real-world units (from calibration)
    pixel_scale = obj_scale

    x = np.arange(0.5, w, 1) * pixel_scale
    y = np.arange(0.5, h, 1) * pixel_scale # matches numpy indexing
    xx, yy = np.meshgrid(x, y)

    mask = d_ind > 0

    # Write vertices
    xyz = np.vstack((xx[mask], yy[mask], d[mask] * pixel_scale)).T
    obj.write(''.join([f"v {x} {y} {z}\n" for x, y, z in xyz]))

    # Build index map
    idx_map = np.zeros_like(d_ind, dtype=np.int32)
    idx_map[mask] = np.arange(1, np.sum(mask) + 1) # 1-indexed for OBJ

    # Create masks for triangle corners
    right = np.roll(mask, -1, axis=1)
    right[:, -1] = 0

    down = np.roll(mask, -1, axis=0)
    down[-1, :] = 0

    right_down = np.roll(right, -1, axis=1)
    right_down[:, -1] = 0

    up_tri = mask & right & down  # counterclockwise
    rows, cols = np.where(up_tri)
    for r, c in zip(rows, cols):
        v1 = idx_map[r, c]
        v2 = idx_map[r, c + 1]
        v3 = idx_map[r + 1, c + 1]
        obj.write(f"f {v1} {v2} {v3}\n")

    low_tri = mask & down & right_down  # counterclockwise
    rows, cols = np.where(low_tri)
    for r, c in zip(rows, cols):
        v1 = idx_map[r, c]
        v2 = idx_map[r + 1, c]
        v3 = idx_map[r + 1, c + 1]
        obj.write(f"f {v1} {v2} {v3}\n")

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
    depth = depth.copy()  # avoid modifying the original array in place
    depth[mask_bg] = np.nan
    np.save(filename, depth)


def remove_polynomial_trend(depth_map, degree=2):
    """
    Given a depth map and degrees will fit a polynomial to the depth map to try and remove noise from image

    :param depth_map: Numpy array of depth map
    :param degree: Degrees of polynomial we are fitting
    :return: Returns the corrected depth map and the fitted polynomial
    """
    # Create a mask for valid (non-NaN) points
    valid_mask = ~np.isnan(depth_map)
    original_shape = depth_map.shape

    # Extract only the valid coordinates and depth values
    y_indices, x_indices = np.where(valid_mask)
    valid_z = depth_map[valid_mask]

    if len(valid_z) < 10:  # Not enough points
        print("WARNING: Not enough valid points for polynomial fitting")
        return depth_map, depth_map, np.zeros(original_shape)

    # Use original coordinates
    x = x_indices.astype(np.float64)
    y = y_indices.astype(np.float64)
    z = valid_z.astype(np.float64)

    # Normalize coordinates
    x_mean, x_std = x.mean(), x.std()
    y_mean, y_std = y.mean(), y.std()
    z_mean, z_std = z.mean(), z.std()

    if x_std < 1e-10 or y_std < 1e-10:
        print("WARNING: Degenerate coordinates for polynomial fitting")
        return depth_map, depth_map, np.zeros(original_shape)

    x_norm = (x - x_mean) / x_std
    y_norm = (y - y_mean) / y_std
    z_norm = (z - z_mean) / z_std

    # Generate polynomial terms
    powers = [(i, j) for i in range(degree + 1) for j in range(degree + 1 - i)]
    A = np.column_stack([x_norm ** i * y_norm ** j for i, j in powers])

    # Fit polynomial using lstsq (faster)
    coeffs, _, _, _ = np.linalg.lstsq(A, z_norm, rcond=None)

    # Build fitted surface on original grid
    X_grid, Y_grid = np.meshgrid(np.arange(original_shape[1]), np.arange(original_shape[0]))

    # Compute fitted values for all points
    x_norm_grid = (X_grid - x_mean) / x_std
    y_norm_grid = (Y_grid - y_mean) / y_std

    fitted_surface = np.zeros(original_shape)
    for k, (i, j) in enumerate(powers):
        fitted_surface += coeffs[k] * (x_norm_grid ** i) * (y_norm_grid ** j)
    fitted_surface = fitted_surface * z_std + z_mean

    # Apply only to valid pixels
    fitted_surface[~valid_mask] = np.nan

    # Subtract fitted surface
    corrected_depth_map = depth_map - fitted_surface

    print(f"Polynomial fitting complete. Degree: {degree}")
    print(f"  Depth range (raw): {np.nanmin(depth_map):.3f} to {np.nanmax(depth_map):.3f}")
    print(f"  Depth range (corrected): {np.nanmin(corrected_depth_map):.3f} to {np.nanmax(corrected_depth_map):.3f}")

    # Return full-sized arrays (same shape as input)
    return depth_map, corrected_depth_map, fitted_surface


def generate_depth_map(normal_path, output, degree=1, depth=None, d_lambda=100, obj_scale=1.0):
    """
    Generate raw and corrected depth maps from normals.

    Args:
        normal_path (str): Path to normal map .npy file
        output (str): Base name for output files
        degree (int): Polynomial degree for trend removal
        depth (str|None): Optional path to depth prior .npy
        d_lambda (float): Weight of depth prior

    Returns:
        tuple: (raw_depth, corrected_depth, fitted_polynomial)
    """
    print("Start reading input data...")
    n, mask = read_normal_map(normal_path)
    mask = mask.astype(bool) # Ensure boolean
    original_shape = mask.shape

    print(f"Normal map shape: {n.shape}")
    print(f"Mask valid pixels: {mask.sum()} / {mask.size}")

    if depth is not None:
        depth = np.load(depth)

    # Compute gradients (p = ∂z/∂x, q = ∂z/∂y)
    p = -n[..., 0] / (n[..., 2] + eps)  # avoid zero division
    q = -n[..., 1] / (n[..., 2] + eps)  # avoid zero division

    # Poisson integration
    print("Start normal integration...")
    task = PoissonOperator(np.dstack([p, q]), mask.astype(np.int8), depth, d_lambda)
    d = task.run()

    print(f"Integration complete. d shape: {d.shape}")
    print(f"d range before masking: {d.min():.3f} to {d.max():.3f}")

    # Apply mask and shift
    d[~mask] = np.nan  # Only invalid pixels become NaN
    d[mask] = d[mask] - d[mask].min()  # Shift min to 0

    print(f"d range after shift: {np.nanmin(d):.3f} to {np.nanmax(d):.3f}")
    print(f"NaN count: {np.isnan(d).sum()}")

    # Polynomial trend removal
    print("Start polynomial fitting...")
    raw_d, c_d, fitted = remove_polynomial_trend(d, degree=degree)

    # Masks should be identical (remove_polynomial_trend preserves NaN)
    raw_mask = ~np.isnan(raw_d)
    c_mask = ~np.isnan(c_d)
    #c_d[c_mask] = c_d[c_mask] - c_d[c_mask].min() # shift corrected to min=0 like raw

    print(f"raw_mask valid pixels: {raw_mask.sum()}, c_mask valid pixels: {c_mask.sum()}")
    print(f"masks identical: {np.array_equal(raw_mask, c_mask)}")
    print(f"raw_d shape: {raw_d.shape}, c_d shape: {c_d.shape}")

    # Export OBJ files for Blender
    print("Start writing output files...")
    write_obj(f"{output}.obj", raw_d, raw_mask, obj_scale=obj_scale)
    write_obj(f"{output}_corrected.obj", c_d, c_mask, obj_scale=obj_scale)

    # Save depth maps as .npy
    write_depth_map(f"{output}.npy", raw_d, ~raw_mask)
    write_depth_map(f"{output}_corrected.npy", c_d, ~c_mask)

    print("Finish!")
    print("Raw depth Z range:   ", np.nanmin(raw_d), "to", np.nanmax(raw_d))
    print("Corrected Z range:   ", np.nanmin(c_d), "to", np.nanmax(c_d))

    return raw_d, c_d, fitted

if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()
    # Prompt user to set directory with images
    directory = filedialog.askdirectory() + '/'
    normal_path = directory + "normals/normal_map.npy"
    output = directory + "depth_map"

    degree = int(input("How many degrees would you like to fit polynomial to: "))

    raw_d, c_d, fitted = generate_depth_map(normal_path, output, degree=degree)

    # Replace NaN with 0 for plotting
    raw_d_plot = np.nan_to_num(raw_d, nan=0.0)
    c_d_plot = np.nan_to_num(c_d, nan=0.0)
    fitted_plot = np.nan_to_num(fitted, nan=0.0)

    print(f"raw_d valid values: {np.sum(~np.isnan(raw_d))} / {raw_d.size}")
    print(f"c_d valid values: {np.sum(~np.isnan(c_d))} / {c_d.size}")
    print(f"fitted valid values: {np.sum(~np.isnan(fitted))} / {fitted.size}")

    # Create figure
    fig1 = go.Figure()
    # Add original depth map
    fig1.add_trace(go.Surface(z=raw_d, colorscale='gray', opacity=0.7, name="Original Depth"))
    # Add fitted polynomial surface
    fig1.add_trace(go.Surface(z=fitted, colorscale='viridis', opacity=0.6, name="Fitted Curve"))
    # Update layout
    fig1.update_layout(title='Original Depth Map vs Fitted Polynomial Surface', autosize=True)
    fig1.show()

    # Display the corrected depth map
    fig2 = go.Figure(data=[go.Surface(z=c_d, colorscale='gray')])
    fig2.update_layout(title='Corrected Depth Map', autosize=True)
    fig2.show()

    # Display original depth map
    fig3 = go.Figure(data=[go.Surface(z=raw_d, colorscale='gray')])
    fig3.update_layout(title='Depth Map', autosize=True)
    fig3.show()
