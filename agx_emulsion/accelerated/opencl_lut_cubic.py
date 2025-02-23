import pyopencl as cl
import numpy as np
from agx_emulsion.accelerated.opencl_context import get_context

def opencl_lut_cubic(lut, image):
    # Prepare context and command queue.
    ctx = get_context()
    queue = cl.CommandQueue(ctx)
    
    # Convert inputs to float32.
    lut = np.array(lut, dtype=np.float32)
    image = np.array(image, dtype=np.float32)
    height, width, _ = image.shape
    L = lut.shape[0]
    
    # Flatten arrays.
    lut_flat = lut.flatten()
    image_flat = image.flatten()
    output = np.empty((height * width * 3), dtype=np.float32)
    mf = cl.mem_flags

    # Create buffers.
    lut_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=lut_flat)
    image_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image_flat)
    output_buf = cl.Buffer(ctx, mf.WRITE_ONLY, output.nbytes)
    
    # OpenCL kernel implementing cubic interpolation via Mitchell–Netravali.
    kernel_code = """
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
         __global const float *image,  // Input image (flattened): H*W*3 floats
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
    
        // Map image pixel to LUT grid coordinates.
        float r = image[img_idx + 0] * (L - 1);
        float g = image[img_idx + 1] * (L - 1);
        float b_val = image[img_idx + 2] * (L - 1);
    
        int r_base = (int)floor(r);
        int g_base = (int)floor(g);
        int b_base = (int)floor(b_val);
    
        float r_frac = r - r_base;
        float g_frac = g - g_base;
        float b_frac = b_val - b_base;
    
        float wr[4], wg[4], wb[4];
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
        float out_r = 0.0f, out_g = 0.0f, out_b = 0.0f;
    
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
    
    # Build and execute the program.
    program = cl.Program(ctx, kernel_code).build()
    global_size = (height * width,)
    program.cubic_lut_interp(
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
    
    cl.enqueue_copy(queue, output, output_buf)
    return output.reshape((height, width, 3))
