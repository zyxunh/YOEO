import torch.distributed as dist

def is_enabled() -> bool:
    """
    Returns:
        True if distributed training is enabled
    """
    return dist.is_available() and dist.is_initialized()


def get_global_size() -> int:
    """
    Returns:
        The number of processes in the process group
    """
    return dist.get_world_size() if is_enabled() else 1


def get_global_rank() -> int:
    """
    Returns:
        The rank of the current process within the global process group.
    """
    return dist.get_rank() if is_enabled() else 0