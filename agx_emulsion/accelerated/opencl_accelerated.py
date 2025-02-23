import pyopencl as cl
import numpy as np
from agx_emulsion.accelerated.opencl_context import get_context, get_queue

_ctx = get_context()
_queue = get_queue()

_gaussian_kernel_source = """
__kernel void gaussian_blur(__global const float* img, __global float* out,
                            const int width, const int height, const float sigma) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    const float effective_sigma = fmax(sigma, 1e-6f);
    const int radius = (int)(3.0f * effective_sigma);
    const int idx = y * width + x;
    
    float sum = 0.0f;
    float weight_sum = 0.0f;
    
    if (radius > 0) {
        for (int dy = -radius; dy <= radius; dy++) {
            // Changed inner loop condition to include upper bound.
            for (int dx = -radius; dx <= radius; dx++) {
                const int nx = clamp(x + dx, 0, width-1);
                const int ny = clamp(y + dy, 0, height-1);
                const float dist2 = dx*dx + dy*dy;
                const float weight = exp(-dist2/(2.0f*effective_sigma*effective_sigma));
                
                sum += img[ny * width + nx] * weight;
                weight_sum += weight;
            }
        }
    } else {
        sum = img[idx];
        weight_sum = 1.0f;
    }
    
    out[idx] = sum / weight_sum;
}
"""

_combine_kernel_source = """
__kernel void combine(__global const float* orig, __global const float* blurred,
                      __global float* out, const int total, const float weight) {
    int i = get_global_id(0);
    if(i < total)
        out[i] = (orig[i] + weight * blurred[i]) / (1.0f + weight);
}
"""

_unsharp_mask_kernel_source = """
__kernel void unsharp_mask(__global const float* orig, __global const float* blurred,
                             __global float* out, const int total, const float amount) {
    int i = get_global_id(0);
    if(i < total) {
        float sharpened = orig[i] + amount * (orig[i] - blurred[i]);
        out[i] = fmin(fmax(sharpened, 0.0f), 1.0f);
    }
}
"""

_prg = cl.Program(_ctx, _gaussian_kernel_source).build()
_combine_prg = cl.Program(_ctx, _combine_kernel_source).build()
_um_prg = cl.Program(_ctx, _unsharp_mask_kernel_source).build()

def _run_gaussian_blur(image_channel, sigma):
    h, w = image_channel.shape
    src = np.ascontiguousarray(image_channel.astype(np.float32))
    dest = np.empty_like(src)
    
    mf = cl.mem_flags
    src_buf = cl.Buffer(_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=src)
    dest_buf = cl.Buffer(_ctx, mf.WRITE_ONLY, dest.nbytes)
    
    safe_sigma = max(float(sigma), 1e-10)
    
    _prg.gaussian_blur(
        _queue, 
        (w, h),
        None, 
        src_buf, 
        dest_buf, 
        np.int32(w), 
        np.int32(h), 
        np.float32(safe_sigma)
    )
    
    cl.enqueue_copy(_queue, dest, dest_buf).wait()
    return dest

def _to_gpu(np_array):
    mf = cl.mem_flags
    buf = cl.Buffer(_ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np_array.astype(np.float32))
    return buf

def _from_gpu(buf, shape):
    np_out = np.empty(shape, dtype=np.float32)
    cl.enqueue_copy(_queue, np_out, buf).wait()
    return np_out

def apply_gaussian_blur_um_gpu(image, sigma_um, pixel_size_um):
    sigma = sigma_um / pixel_size_um
    return apply_gaussian_blur_gpu(image, sigma)

def apply_halation_um_gpu(image, halation, pixel_size_um):
    out = image.copy()
    for i in range(image.shape[2]):
        if (halation.strength[i] > 0):
            sigma = halation.size_um[i] / pixel_size_um
            blurred = _run_gaussian_blur(image[:, :, i], sigma)
            out[:, :, i] = (image[:, :, i] + halation.strength[i] * blurred) / (1.0 + halation.strength[i])
    for i in range(image.shape[2]):
        if (halation.scattering_strength[i] > 0):
            sigma = halation.scattering_size_um[i] / pixel_size_um
            blurred = _run_gaussian_blur(image[:, :, i], sigma)
            out[:, :, i] = (out[:, :, i] + halation.scattering_strength[i] * blurred) / (1.0 + halation.scattering_strength[i])
    return out

def apply_gaussian_blur_gpu(image, sigma):
    # Copy image to GPU once
    h, w, c = image.shape
    in_buf = _to_gpu(image)
    out_buf = cl.Buffer(_ctx, cl.mem_flags.READ_WRITE, image.nbytes)

    # Run blur for each channel without copying back every time
    for ch in range(c):
        _prg.gaussian_blur(
            _queue,
            (w, h),
            None,
            in_buf,
            out_buf,
            np.int32(w),
            np.int32(h),
            np.float32(max(float(sigma), 1e-10))
        )
        # Swap buffers so next channel uses blurred data
        tmp = in_buf
        in_buf = out_buf
        out_buf = tmp

    # Done computing, copy back once
    return _from_gpu(in_buf, image.shape)

def apply_unsharp_mask_gpu(image, sigma, amount):
    blurred = apply_gaussian_blur_gpu(image, sigma)
    sharpened = np.clip(image + amount * (image - blurred), 0, 1)
    return sharpened

def apply_combined_halation_gaussian_gpu(image, sigma, halation, pixel_size_um):
    h, w, channels = image.shape
    mf = cl.mem_flags
    final = np.empty_like(image, dtype=np.float32)
    total = np.int32(w * h)
    
    for ch in range(channels):
        chan = np.ascontiguousarray(image[:, :, ch].astype(np.float32))
        src_buf = cl.Buffer(_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=chan)
        tmp_buf = cl.Buffer(_ctx, mf.READ_WRITE, chan.nbytes)
        
        if halation.strength[ch] > 0:
            sigma_h = halation.size_um[ch] / pixel_size_um
            _prg.gaussian_blur(
                _queue,
                (w, h),
                None,
                src_buf,
                tmp_buf,
                np.int32(w),
                np.int32(h),
                np.float32(max(sigma_h, 1e-10))
            )
            combined_buf = cl.Buffer(_ctx, mf.READ_WRITE, chan.nbytes)
            _combine_prg.combine(
                _queue,
                (total,),
                None,
                src_buf,
                tmp_buf,
                combined_buf,
                total,
                np.float32(halation.strength[ch])
            )
            src_buf = combined_buf
        
        if halation.scattering_strength[ch] > 0:
            sigma_s = halation.scattering_size_um[ch] / pixel_size_um
            tmp2_buf = cl.Buffer(_ctx, mf.READ_WRITE, chan.nbytes)
            _prg.gaussian_blur(
                _queue,
                (w, h),
                None,
                src_buf,
                tmp2_buf,
                np.int32(w),
                np.int32(h),
                np.float32(max(sigma_s, 1e-10))
            )
            combined2_buf = cl.Buffer(_ctx, mf.READ_WRITE, chan.nbytes)
            _combine_prg.combine(
                _queue,
                (total,),
                None,
                src_buf,
                tmp2_buf,
                combined2_buf,
                total,
                np.float32(halation.scattering_strength[ch])
            )
            src_buf = combined2_buf
        
        dest_buf = cl.Buffer(_ctx, mf.WRITE_ONLY, chan.nbytes)
        _prg.gaussian_blur(
            _queue,
            (w, h),
            None,
            src_buf,
            dest_buf,
            np.int32(w),
            np.int32(h),
            np.float32(max(sigma, 1e-10))
        )
        chan_out = np.empty_like(chan)
        cl.enqueue_copy(_queue, chan_out, dest_buf).wait()
        final[:, :, ch] = chan_out
    
    return final

def apply_combined_gaussian_unsharp_gpu(image, sigma_pre, sigma_final, amount):
    """
    Apply combined gaussian blur with two sigma values and unsharp mask on GPU.
    Process each channel separately.
    """
    h, w, channels = image.shape
    mf = cl.mem_flags
    final = np.empty_like(image, dtype=np.float32)
    total = np.int32(w * h)
    
    for ch in range(channels):
        # Process each channel separately
        chan = np.ascontiguousarray(image[:, :, ch].astype(np.float32))
        orig_chan_buf = cl.Buffer(_ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=chan)
        
        # First gaussian blur pass with sigma_pre:
        tmp_buf = cl.Buffer(_ctx, mf.READ_WRITE, chan.nbytes)
        _prg.gaussian_blur(
            _queue,
            (w, h),
            None,
            orig_chan_buf,
            tmp_buf,
            np.int32(w),
            np.int32(h),
            np.float32(max(sigma_pre, 1e-10))
        )
        
        # Second gaussian blur pass with sigma_final:
        blurred_buf = cl.Buffer(_ctx, mf.READ_WRITE, chan.nbytes)
        _prg.gaussian_blur(
            _queue,
            (w, h),
            None,
            tmp_buf,
            blurred_buf,
            np.int32(w),
            np.int32(h),
            np.float32(max(sigma_final, 1e-10))
        )
        
        # Apply unsharp mask: final = clip(orig + amount*(orig - blurred))
        dest_buf = cl.Buffer(_ctx, mf.WRITE_ONLY, chan.nbytes)
        _um_prg.unsharp_mask(
            _queue,
            (total,),
            None,
            orig_chan_buf,
            blurred_buf,
            dest_buf,
            total,
            np.float32(amount)
        )
        
        chan_out = np.empty_like(chan)
        cl.enqueue_copy(_queue, chan_out, dest_buf).wait()
        final[:, :, ch] = chan_out

    return final