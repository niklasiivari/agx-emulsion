# Updated contract function
import numpy as np
import pyopencl as cl
from agx_emulsion.accelerated.opencl_context import get_context, get_queue
from agx_emulsion.accelerated.contract_kernel import get_contract_kernel, get_contract_kernel_ijkl

def opencl_parallel_contract(pattern, aces, aces_conversion_matrix):
    """
    GPU-accelerated contraction using OpenCL.
    Supports patterns "ijk, kl->ijl", "ijk,lk->ijl", "k,kl->l" and "ijkl,lm->ijkm".
    """
    aces = np.ascontiguousarray(aces, dtype=np.float32)
    M = np.ascontiguousarray(aces_conversion_matrix, dtype=np.float32)

    # Matrix ordering and contiguity handling
    if pattern in ('ijk, kl->ijl', 'k,kl->l'):
        M_used = np.ascontiguousarray(M.T, dtype=np.float32)  # Force C-contiguous
    elif pattern == 'ijk,lk->ijl':
        M_used = np.ascontiguousarray(M, dtype=np.float32)
    elif pattern == 'ijkl,lm->ijkm':
        M_used = np.ascontiguousarray(M, dtype=np.float32)
    else:
        from opt_einsum import contract
        return contract(pattern, aces, aces_conversion_matrix)

    # Kernel dispatch
    if pattern in ('ijk, kl->ijl', 'ijk,lk->ijl'):
        height, width, input_channels = aces.shape
        output_channels = M_used.shape[0]
        
        # Ensure contiguous memory layout
        a_flat = np.ascontiguousarray(aces.reshape(-1, input_channels))
        
        ctx = get_context()
        queue = get_queue()
        mf = cl.mem_flags

        a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_flat)
        M_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=M_used)
        result = np.empty(height * width * output_channels, dtype=np.float32)
        res_buf = cl.Buffer(ctx, mf.WRITE_ONLY, result.nbytes)

        get_contract_kernel()(queue, (height * width,), None,
                             a_buf, M_buf, res_buf,
                             np.int32(input_channels), np.int32(output_channels))
        
        cl.enqueue_copy(queue, result, res_buf).wait()
        return result.reshape(height, width, output_channels)

    elif pattern == 'k,kl->l':
        input_channels = aces.shape[0]
        output_channels = M_used.shape[0]
        a_flat = np.ascontiguousarray(aces.reshape(1, -1))

        ctx = get_context()
        queue = get_queue()
        mf = cl.mem_flags

        a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_flat)
        M_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=M_used)
        result = np.empty(output_channels, dtype=np.float32)
        res_buf = cl.Buffer(ctx, mf.WRITE_ONLY, result.nbytes)

        get_contract_kernel()(queue, (1,), None,
                             a_buf, M_buf, res_buf,
                             np.int32(input_channels), np.int32(output_channels))
        
        cl.enqueue_copy(queue, result, res_buf).wait()
        return result

    elif pattern == 'ijkl,lm->ijkm':
        i, j, k, L = aces.shape
        M_dim = M.shape[1]
        a_flat = np.ascontiguousarray(aces.reshape(-1, L))

        ctx = get_context()
        queue = get_queue()
        mf = cl.mem_flags

        a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_flat)
        M_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=M_used)
        result = np.empty(i * j * k * M_dim, dtype=np.float32)
        res_buf = cl.Buffer(ctx, mf.WRITE_ONLY, result.nbytes)

        get_contract_kernel_ijkl()(queue, (i * j * k,), None,
                                  a_buf, M_buf, res_buf,
                                  np.int32(L), np.int32(M_dim))
        
        cl.enqueue_copy(queue, result, res_buf).wait()
        return result.reshape(i, j, k, M_dim)

    else:
        from opt_einsum import contract
        return contract(pattern, aces, aces_conversion_matrix)