import pyopencl as cl
import threading

_lock = threading.Lock()
_ctx = None
_program_cache = {}

def get_context():
    global _ctx
    with _lock:
        if _ctx is None:
            _ctx = cl.create_some_context()
        return _ctx

def get_queue():
    return cl.CommandQueue(get_context())

def get_program(kernel_code):
    key = kernel_code
    if key not in _program_cache:
        _program_cache[key] = cl.Program(get_context(), kernel_code).build()
    return _program_cache[key]
