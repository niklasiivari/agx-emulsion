import numpy as np
import pyopencl as cl
from agx_emulsion.accelerated.opencl_context import get_context, get_queue

_ctx = get_context()
_queue = get_queue()

def opencl_create_lut3d(function, xmin=0.0, xmax=1.0, steps=32):
    """
    GPU-accelerated version that exactly replicates:
    def _create_lut3d(function, xmin=0, xmax=1, steps=32):
        x = np.linspace(xmin, xmax, steps, endpoint=True)
        X = np.meshgrid(x,x,x, indexing='ij')
        X = np.stack(X, axis=3)
        X = np.reshape(X, (steps**3, 1, 3))
        lut = np.reshape(function(X), (steps, steps, steps, 3))
        return lut
    """
    # Kernel code that exactly replicates NumPy's meshgrid+stack+reshape
    kernel = """//CL//
    __kernel void generate_grid(__global float* output,
                                float xmin,
                                float xmax,
                                int steps) {
        int idx = get_global_id(0);
        int total = steps * steps * steps;
        if(idx >= total) return;
        
        // Replicate np.meshgrid(x,x,x, indexing='ij')
        int i = idx / (steps * steps);  // x-dimension
        int j = (idx / steps) % steps;  // y-dimension
        int k = idx % steps;            // z-dimension
        
        // Replicate np.linspace(xmin, xmax, steps, endpoint=True)
        float spacing = (xmax - xmin) / (steps - 1.0f);
        float x_val = xmin + i * spacing;
        float y_val = xmin + j * spacing;
        float z_val = xmin + k * spacing;
        
        // Replicate np.stack(X, axis=3) then reshape to (steps^3, 1, 3)
        int pos = idx * 3;
        output[pos] = x_val;    // Channel 0
        output[pos+1] = y_val;  // Channel 1
        output[pos+2] = z_val;  // Channel 2
    }
    """
    
    # Create OpenCL program
    program = cl.Program(_ctx, kernel).build()
    
    # Allocate buffers
    total_points = steps ** 3
    buffer_size = total_points * 3 * 4  # 3 channels * float32
    grid_buffer = cl.Buffer(_ctx, cl.mem_flags.WRITE_ONLY, buffer_size)
    
    # Execute kernel
    program.generate_grid(
        _queue, (total_points,), None,
        grid_buffer,
        np.float32(xmin),
        np.float32(xmax),
        np.int32(steps)
    ).wait()
    
    # Copy results back and reshape exactly like NumPy
    grid_flat = np.empty(total_points * 3, dtype=np.float32)
    cl.enqueue_copy(_queue, grid_flat, grid_buffer).wait()
    
    # Reshape to (steps^3, 1, 3) to match original processing
    grid_reshaped = grid_flat.reshape(total_points, 1, 3)
    
    # Apply Python function to the grid
    lut = function(grid_reshaped)
    
    # Final reshape to (steps, steps, steps, 3)
    return lut.reshape(steps, steps, steps, 3)