import numpy as np
from numba import njit, prange
from scipy.ndimage import map_coordinates
import time
import matplotlib.pyplot as plt
import pyopencl as cl

@njit(cache=True)
def mitchell_weight(t, B=1/3, C=1/3):
    """
    Computes the Mitchell–Netravali kernel weight.
    Parameters:
      t : distance (can be negative)
      B, C : parameters (common choice: B = 1/3, C = 1/3)
    Returns:
      The kernel weight.
    """
    x = abs(t)
    if x < 1:
        return (1/6)*((12 - 9*B - 6*C)*x**3 + (-18 + 12*B + 6*C)*x**2 + (6 - 2*B))
    elif x < 2:
        return (1/6)*((-B - 6*C)*x**3 + (6*B + 30*C)*x**2 + (-12*B - 48*C)*x + (8*B + 24*C))
    else:
        return 0.0

@njit(cache=True)
def safe_index(idx, L):
    """
    Reflects an index idx into the valid range [0, L-1] using symmetric reflection.
    """
    if idx < 0:
        return -idx
    elif idx >= L:
        return 2*(L - 1) - idx
    else:
        return idx

@njit(cache=True)
def cubic_interp_lut_at(lut, r, g, b):
    """
    Performs cubic interpolation at a single point (r, g, b) in LUT space using the
    Mitchell–Netravali kernel.
    
    Parameters:
      - lut: 4D LUT array of shape (L, L, L, 3) (double precision).
             The LUT input domain is assumed to be normalized in [0, 1] (mapped to [0, L-1]),
             while the LUT output values can be any double.
      - r, g, b: Coordinates (floats) in [0, L-1] in LUT space.
    
    Returns:
      - Interpolated RGB value (3-element vector) as double.
    """
    L = lut.shape[0]
    
    # Compute base indices and fractional parts.
    r_base = int(np.floor(r))
    g_base = int(np.floor(g))
    b_base = int(np.floor(b))
    r_frac = r - r_base
    g_frac = g - g_base
    b_frac = b - b_base

    # Compute kernel weights for the 4 neighboring indices along each dimension.
    wr = np.empty(4, dtype=np.float64)
    wg = np.empty(4, dtype=np.float64)
    wb = np.empty(4, dtype=np.float64)
    wr[0] = mitchell_weight(r_frac + 1)
    wr[1] = mitchell_weight(r_frac)
    wr[2] = mitchell_weight(r_frac - 1)
    wr[3] = mitchell_weight(r_frac - 2)
    
    wg[0] = mitchell_weight(g_frac + 1)
    wg[1] = mitchell_weight(g_frac)
    wg[2] = mitchell_weight(g_frac - 1)
    wg[3] = mitchell_weight(g_frac - 2)
    
    wb[0] = mitchell_weight(b_frac + 1)
    wb[1] = mitchell_weight(b_frac)
    wb[2] = mitchell_weight(b_frac - 1)
    wb[3] = mitchell_weight(b_frac - 2)
    
    # Accumulate weighted sum and total weight.
    out = np.zeros(3, dtype=np.float64)
    weight_sum = 0.0
    for i in range(4):
        ri = safe_index(r_base - 1 + i, L)
        for j in range(4):
            gj = safe_index(g_base - 1 + j, L)
            for k in range(4):
                bk = safe_index(b_base - 1 + k, L)
                weight = wr[i] * wg[j] * wb[k]
                weight_sum += weight
                out[0] += weight * lut[ri, gj, bk, 0]
                out[1] += weight * lut[ri, gj, bk, 1]
                out[2] += weight * lut[ri, gj, bk, 2]
    if weight_sum != 0.0:
        out[0] /= weight_sum
        out[1] /= weight_sum
        out[2] /= weight_sum
    return out

@njit(parallel=True, cache=True)
def apply_lut_cubic(lut, image):
    """
    Applies a 3D LUT (shape: 32×32×32×3) to an image (shape: H×W×3)
    using cubic interpolation with the Mitchell–Netravali kernel.
    
    Assumptions:
      - The input image values and the LUT input domain are normalized in [0,1].
      - The LUT is defined on a grid from [0, L-1] in each dimension.
      - The LUT's output values are double (and can be unbounded).
    
    Returns:
      - The output image (H×W×3) with the applied LUT (double precision).
    """
    height, width, _ = image.shape
    output = np.empty((height, width, 3), dtype=np.float64)
    L = lut.shape[0]
    for i in prange(height):
        for j in range(width):
            r_in = image[i, j, 0] * (L - 1)
            g_in = image[i, j, 1] * (L - 1)
            b_in = image[i, j, 2] * (L - 1)
            out_val = cubic_interp_lut_at(lut, r_in, g_in, b_in)
            output[i, j, 0] = out_val[0]
            output[i, j, 1] = out_val[1]
            output[i, j, 2] = out_val[2]
    return output

def apply_lut_cubic_scipy(lut, image):
    """
    Reference implementation using SciPy's ndimage.map_coordinates.
    Applies cubic interpolation (order=3) for each channel of the LUT with reflection mode.
    
    Parameters:
      - lut: 4D LUT array (L, L, L, 3) with double precision.
      - image: Input image array (H, W, 3) with values in [0,1].
    
    Returns:
      - Output image (H, W, 3) with the applied LUT (double precision).
    """
    height, width, _ = image.shape
    L = lut.shape[0]
    coords = np.empty((3, height, width), dtype=np.float64)
    coords[0] = image[:, :, 0] * (L - 1)
    coords[1] = image[:, :, 1] * (L - 1)
    coords[2] = image[:, :, 2] * (L - 1)
    output = np.empty((height, width, 3), dtype=np.float64)
    for c in range(3):
        output[:, :, c] = map_coordinates(lut[..., c], coords, order=3, mode='reflect')
    return output

def opencl_apply_lut_cubic(lut, image):
    """
    GPU-accelerated cubic LUT interpolation using OpenCL and the Mitchell–Netravali kernel.
    
    Parameters:
      lut   : 4D LUT array of shape (L, L, L, 3) (float32) where input domain is [0,1] scaled to [0, L-1].
      image : Input image array of shape (H, W, 3) (float32) with values in [0,1].
    
    Returns:
      Output image (H, W, 3) after LUT interpolation (float32).
    """
    # Convert inputs to float32 for GPU efficiency.
    lut = np.array(lut, dtype=np.float32)
    image = np.array(image, dtype=np.float32)
    height, width, _ = image.shape
    L = lut.shape[0]
    
    # Flatten arrays.
    lut_flat = lut.flatten()
    image_flat = image.flatten()
    output = np.empty((height * width * 3), dtype=np.float32)
    
    # Create OpenCL context and command queue.
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    
    # Create OpenCL buffers.
    lut_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=lut_flat)
    image_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image_flat)
    output_buf = cl.Buffer(ctx, mf.WRITE_ONLY, output.nbytes)
    
    # OpenCL kernel implementing cubic interpolation using the Mitchell–Netravali kernel.
    kernel_code = """
    #pragma OPENCL EXTENSION cl_khr_fp32 : enable
    inline float mitchell_weight(float t, float B, float C) {
        float x = fabs(t);
        if (x < 1.0f) {
            return (1.0f/6.0f)*((12.0f - 9.0f*B - 6.0f*C)*x*x*x +
                                 (-18.0f + 12.0f*B + 6.0f*C)*x*x +
                                 (6.0f - 2.0f*B));
        } else if (x < 2.0f) {
            return (1.0f/6.0f)*((-B - 6.0f*C)*x*x*x +
                                 (6.0f*B + 30.0f*C)*x*x +
                                 (-12.0f*B - 48.0f*C)*x +
                                 (8.0f*B + 24.0f*C));
        } else {
            return 0.0f;
        }
    }
    
    inline int safe_index(int idx, int L) {
        if (idx < 0)
            return -idx;
        else if (idx >= L)
            return 2*(L - 1) - idx;
        else
            return idx;
    }
    
    __kernel void cubic_lut_interp(
         __global const float *lut,    // LUT (flattened): L*L*L*3 floats
         __global const float *image,  // Image (flattened): H*W*3 floats
         __global float *output,       // Output image (flattened)
         const int L,                  // LUT grid size
         const int width,              // Image width
         const int height,             // Image height
         const float B,                // Mitchell–Netravali parameter B
         const float C)                // Mitchell–Netravali parameter C
    {
        int gid = get_global_id(0);
        if (gid >= height * width)
            return;
            
        int i = gid / width;
        int j = gid % width;
        int img_idx = (i * width + j) * 3;
    
        // Map image pixel values to LUT grid coordinates.
        float r = image[img_idx + 0] * (L - 1);
        float g = image[img_idx + 1] * (L - 1);
        float b = image[img_idx + 2] * (L - 1);
    
        int r_base = (int)floor(r);
        int g_base = (int)floor(g);
        int b_base = (int)floor(b);
    
        float r_frac = r - r_base;
        float g_frac = g - g_base;
        float b_frac = b - b_base;
    
        // Precompute weights for each dimension.
        float wr[4];
        float wg[4];
        float wb[4];
    
        wr[0] = mitchell_weight(r_frac + 1.0f, B, C);
        wr[1] = mitchell_weight(r_frac,       B, C);
        wr[2] = mitchell_weight(r_frac - 1.0f,  B, C);
        wr[3] = mitchell_weight(r_frac - 2.0f,  B, C);
    
        wg[0] = mitchell_weight(g_frac + 1.0f, B, C);
        wg[1] = mitchell_weight(g_frac,       B, C);
        wg[2] = mitchell_weight(g_frac - 1.0f,  B, C);
        wg[3] = mitchell_weight(g_frac - 2.0f,  B, C);
    
        wb[0] = mitchell_weight(b_frac + 1.0f, B, C);
        wb[1] = mitchell_weight(b_frac,       B, C);
        wb[2] = mitchell_weight(b_frac - 1.0f,  B, C);
        wb[3] = mitchell_weight(b_frac - 2.0f,  B, C);
    
        float weight_sum = 0.0f;
        float out_r = 0.0f;
        float out_g = 0.0f;
        float out_b = 0.0f;
    
        // Accumulate weighted contributions from the 4x4x4 neighborhood.
        for (int di = 0; di < 4; di++) {
            int ri = safe_index(r_base - 1 + di, L);
            for (int dj = 0; dj < 4; dj++) {
                int gi = safe_index(g_base - 1 + dj, L);
                for (int dk = 0; dk < 4; dk++) {
                    int bi = safe_index(b_base - 1 + dk, L);
                    float w = wr[di] * wg[dj] * wb[dk];
                    weight_sum += w;
                    int lut_idx = ((ri * L + gi) * L + bi) * 3;
                    out_r += w * lut[lut_idx + 0];
                    out_g += w * lut[lut_idx + 1];
                    out_b += w * lut[lut_idx + 2];
                }
            }
        }
        
        // Normalize the output color.
        if (weight_sum > 0.0f) {
            out_r /= weight_sum;
            out_g /= weight_sum;
            out_b /= weight_sum;
        }
    
        output[img_idx + 0] = out_r;
        output[img_idx + 1] = out_g;
        output[img_idx + 2] = out_b;
    }
    """
    # Build the OpenCL program.
    prg = cl.Program(ctx, kernel_code).build()

    global_size = (height * width,)
    # Call the kernel with the full list of required arguments:
    prg.cubic_lut_interp(
         queue,
         global_size,
         None,
         lut_buf,
         image_buf,
         output_buf,
         np.int32(L),
         np.int32(width),
         np.int32(height),
         np.float32(1.0/3.0),
         np.float32(1.0/3.0)
    )
    
    # Retrieve the result.
    cl.enqueue_copy(queue, output, output_buf)
    return output.reshape((height, width, 3))

if __name__ == '__main__':
    # --- Create a Synthetic LUT ---
    # Here we create a LUT that applies a simple non-linear transformation.
    # For clarity, the LUT output is defined as (r^2, g^2, b^2) for each channel.
    L = 32
    grid = np.linspace(0, 1, L, dtype=np.float64)
    R, G, B = np.meshgrid(grid, grid, grid, indexing='ij')
    lut = np.stack((R**2, G**2, B**2), axis=-1)  # double precision; unbounded output
    
    # --- Create a Synthetic Test Image ---
    # Generate a clear gradient image:
    # - Red channel varies horizontally,
    # - Green channel varies vertically,
    # - Blue channel is fixed at 0.5.
    height, width = 512, 512
    x = np.linspace(0, 1, width, dtype=np.float64)
    y = np.linspace(0, 1, height, dtype=np.float64)
    X, Y = np.meshgrid(x, y)
    image = np.stack((X, Y, 0.5 * np.ones_like(X)), axis=-1)  # double precision
    
    # --- Warm Up the JIT Compiler ---
    _ = apply_lut_cubic(lut, image)
    
    # --- Benchmark the Numba Implementation ---
    iterations = 10
    start_time = time.time()
    for _ in range(iterations):
        output_numba = apply_lut_cubic(lut, image)
    numba_time = (time.time() - start_time) / iterations
    print("Average time per iteration (Numba cubic interpolation): {:.6f} seconds".format(numba_time))
    
    # --- Benchmark the SciPy Reference Implementation ---
    start_time = time.time()
    for _ in range(iterations):
        output_scipy = apply_lut_cubic_scipy(lut, image)
    scipy_time = (time.time() - start_time) / iterations
    print("Average time per iteration (SciPy cubic interpolation): {:.6f} seconds".format(scipy_time))
    
    # --- Benchmark the OpenCL Implementation ---
    start_time = time.time()
    for _ in range(iterations):
        output_opencl = opencl_apply_lut_cubic(lut, image)
    opencl_time = (time.time() - start_time) / iterations
    print("Average time per iteration (OpenCL cubic interpolation): {:.6f} seconds".format(opencl_time))
    
    # --- Compute Error Metrics ---
    diff_numba_scipy = output_numba - output_scipy
    rmse_numba_scipy = np.sqrt(np.mean(diff_numba_scipy**2))
    max_error_numba_scipy = np.max(np.abs(diff_numba_scipy))
    print("RMSE error between Numba and SciPy outputs: {:.6e}".format(rmse_numba_scipy))
    print("Max absolute error between Numba and SciPy outputs: {:.6e}".format(max_error_numba_scipy))
    
    diff_numba_opencl = output_numba - output_opencl
    rmse_numba_opencl = np.sqrt(np.mean(diff_numba_opencl**2))
    max_error_numba_opencl = np.max(np.abs(diff_numba_opencl))
    print("RMSE error between Numba and OpenCL outputs: {:.6e}".format(rmse_numba_opencl))
    print("Max absolute error between Numba and OpenCL outputs: {:.6e}".format(max_error_numba_opencl))
    
    # --- Improved Plotting ---
    # Compute an error map as the per-pixel L2 norm of the difference.
    diff_norm = np.sqrt(np.sum(diff_numba_scipy**2, axis=2))
    
    # Create a 2x2 subplot layout.
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    
    # Input image (gradient).
    axs[0, 0].imshow(image, interpolation='nearest')
    axs[0, 0].set_title("Input Gradient Image")
    axs[0, 0].axis("off")
    
    # Output from the Numba implementation.
    axs[0, 1].imshow(output_numba, interpolation='nearest')
    axs[0, 1].set_title("Output (Numba, Mitchell–Netravali)")
    axs[0, 1].axis("off")
    
    # Output from the SciPy implementation.
    axs[1, 0].imshow(output_scipy, interpolation='nearest')
    axs[1, 0].set_title("Output (SciPy, Reflect Mode)")
    axs[1, 0].axis("off")
    
    # Error map (L2 norm per pixel) with a colorbar.
    im = axs[1, 1].imshow(diff_norm, cmap="hot", interpolation="nearest")
    axs[1, 1].set_title("Error Map (L2 Norm)")
    axs[1, 1].axis("off")
    fig.colorbar(im, ax=axs[1, 1], fraction=0.046, pad=0.04)
    
    fig.suptitle("LUT Cubic Interpolation Comparison (Double Precision, Improved Kernel)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
