"""Peak resident memory of the current process, on Windows and Linux.

Pipeline position: helper used by the stages to log peak memory.
"""

import sys


def peak_mem_gb():
    """Peak memory of this process in GB (peak working set on Windows, max RSS on Linux)."""
    if sys.platform == "win32":
        import psutil
        return psutil.Process().memory_info().peak_wset / 1024 ** 3
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2  # KB on Linux
