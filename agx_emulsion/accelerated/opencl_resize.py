import pyopencl as cl
import numpy as np
from agx_emulsion.accelerated.opencl_context import get_context, get_queue

_ctx = get_context()
_queue = get_queue()

_resize_kernel_source = """
__kernel void resize_bilinear(
    __global const float* input,
    __global float* output,
    const int src_width,
    const int src_height,
    const int channels,
    const int dst_width,
    const int dst_height) {
    
    int x = get_global_id(0);
    int y = get_global_id(1);
    if(x >= dst_width || y >= dst_height) return;
    
    float scale_x = (float)src_width / dst_width;
    float scale_y = (float)src_height / dst_height;
    
    float fx = x * scale_x;
    float fy = y * scale_y;
    int x0 = (int)floor(fx);
    int y0 = (int)floor(fy);
    int x1 = min(x0 + 1, src_width - 1);
    int y1 = min(y0 + 1, src_height - 1);
    float wx = fx - x0;
    float wy = fy - y0;
    
    for(int c = 0; c < channels; c++){
        float v00 = input[(y0 * src_width + x0) * channels + c];
        float v10 = input[(y0 * src_width + x1) * channels + c];
        float v01 = input[(y1 * src_width + x0) * channels + c];
        float v11 = input[(y1 * src_width + x1) * channels + c];
        float v0 = v00 + wx * (v10 - v00);
        float v1 = v01 + wx * (v11 - v01);
        float value = v0 + wy * (v1 - v0);
        output[(y * dst_width + x) * channels + c] = value;
    }
}
"""

_pr = cl.Program(_ctx, _resize_kernel_source).build()

def resize_image_gpu(image, resize_factor):
    """
    Resize the given image using bilinear interpolation on GPU.
    image: numpy.ndarray of shape (H, W, C) as float32.
    resize_factor: float scaling factor.
    Returns:
      Resized image as numpy.ndarray.
    """
    src_h, src_w, channels = image.shape
    dst_w = int(src_w * resize_factor)
    dst_h = int(src_h * resize_factor)
    
    mf = cl.mem_flags
    input_buf = cl.Buffer(_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image.astype(np.float32))
    output = np.empty((dst_h, dst_w, channels), dtype=np.float32)
    output_buf = cl.Buffer(_ctx, mf.WRITE_ONLY, output.nbytes)
    
    global_size = (dst_w, dst_h)
    _pr.resize_bilinear(_queue, global_size, None,
                         input_buf,
                         output_buf,
                         np.int32(src_w),
                         np.int32(src_h),
                         np.int32(channels),
                         np.int32(dst_w),
                         np.int32(dst_h))
    
    cl.enqueue_copy(_queue, output, output_buf).wait()
    return output
