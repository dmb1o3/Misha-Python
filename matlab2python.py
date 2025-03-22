import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr


def depth_map(surf_normals, mask_image):
    """
    Convert surface normals to a depth map using a least squares approach.

    Parameters:
    surf_normals: numpy array of shape (height, width, 3) containing surface normal vectors
    mask_image: boolean numpy array of shape (height, width) indicating object pixels

    Returns:
    z: numpy array of shape (height, width) containing the depth map
    """
    # Get dimensions
    nrows, ncols = mask_image.shape
    print(f"nrows = {nrows}")
    print(f"ncols = {ncols}")

    # Find object pixels
    object_pixels = np.argwhere(mask_image)

    # Create index matrix to quickly retrieve pixel indices
    index = np.zeros((nrows, ncols), dtype=int)

    # Total number of pixels within the mask
    num_pixels = len(object_pixels)
    print(f"num_pixels = {num_pixels}")

    # Assign indices to object pixels (using 0-indexing)
    for d in range(num_pixels):
        p_row, p_col = object_pixels[d]
        index[p_row, p_col] = d  # Store 0-indexed position

    # Create sparse matrix M and vector b
    M = sparse.lil_matrix((2 * num_pixels, num_pixels))
    b = np.zeros(2 * num_pixels)

    print('entering depth map loop')
    for d in range(num_pixels):
        p_row, p_col = object_pixels[d]
        nx = surf_normals[p_row, p_col, 0]
        ny = surf_normals[p_row, p_col, 1]
        nz = surf_normals[p_row, p_col, 2]

        # Both (X+1, Y) and (X, Y+1) are inside the object
        if p_col + 1 < ncols and p_row - 1 >= 0 and index[p_row, p_col + 1] > 0 and index[p_row - 1, p_col] > 0:
            M[2 * d, index[p_row, p_col]] = 1
            M[2 * d, index[p_row, p_col + 1]] = -1  # (X+1, Y)
            b[2 * d] = nx / nz

            M[2 * d + 1, index[p_row, p_col]] = 1
            M[2 * d + 1, index[p_row - 1, p_col]] = -1  # (X, Y+1)
            b[2 * d + 1] = ny / nz

        # (X, Y+1) is inside but (X+1, Y) is outside
        elif p_row - 1 >= 0 and index[p_row - 1, p_col] > 0:
            f = -1
            if p_col + f >= 0 and p_col + f < ncols and index[p_row, p_col + f] > 0:
                M[2 * d, index[p_row, p_col]] = 1
                M[2 * d, index[p_row, p_col + f]] = -1  # (X+f, Y)
                b[2 * d] = f * nx / nz

            M[2 * d + 1, index[p_row, p_col]] = 1
            M[2 * d + 1, index[p_row - 1, p_col]] = -1  # (X, Y+1)
            b[2 * d + 1] = ny / nz

        # (X+1, Y) is inside but (X, Y+1) is outside
        elif p_col + 1 < ncols and index[p_row, p_col + 1] > 0:
            f = -1
            if p_row - f >= 0 and p_row - f < nrows and index[p_row - f, p_col] > 0:
                M[2 * d + 1, index[p_row, p_col]] = 1
                M[2 * d + 1, index[p_row - f, p_col]] = -1  # (X, Y+f)
                b[2 * d + 1] = f * ny / nz

            M[2 * d, index[p_row, p_col]] = 1
            M[2 * d, index[p_row, p_col + 1]] = -1  # (X+1, Y)
            b[2 * d] = nx / nz

        # Both (X+1, Y) and (X, Y+1) are outside
        else:
            f = -1
            if p_col + f >= 0 and p_col + f < ncols and index[p_row, p_col + f] > 0:
                M[2 * d, index[p_row, p_col]] = 1
                M[2 * d, index[p_row, p_col + f]] = -1  # (X+f, Y)
                b[2 * d] = f * nx / nz

            f = -1
            if p_row - f >= 0 and p_row - f < nrows and index[p_row - f, p_col] > 0:
                M[2 * d + 1, index[p_row, p_col]] = 1
                M[2 * d + 1, index[p_row - f, p_col]] = -1  # (X, Y+f)
                b[2 * d + 1] = f * ny / nz

    # Convert to CSR format for efficient solving
    M = M.tocsr()

    # Solve the overdetermined system using least squares (equivalent to MATLAB's backslash)
    print("Solving the linear system...")
    x = lsqr(M, b)[0]  # Get just the solution vector
    x = x - np.min(x)

    # Populate the depth map
    temp_shape = np.zeros((nrows, ncols))
    for d in range(num_pixels):
        p_row, p_col = object_pixels[d]
        temp_shape[p_row, p_col] = x[d]

    # Flip the depth map to match MATLAB's orientation
    z = np.zeros((nrows, ncols))
    for i in range(nrows):
        for j in range(ncols):
            z[i, j] = temp_shape[nrows - i - 1, j]

    return z
