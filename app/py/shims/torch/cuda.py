"""torch.cuda: there is no CUDA in a browser."""


class OutOfMemoryError(RuntimeError):
    pass


def is_available():
    return False


def empty_cache():
    pass
