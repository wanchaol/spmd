from distributed.tensor import (
    Tensor
)

def unwrap_single_placement(e):
    if not isinstance(e, Tensor):
        return None
    assert len(e.placements) == 1, "more than one placement!"
    return e.placements[0]

def unwrap_local_tensor(e):
    if not isinstance(e, Tensor):
        return None
    return e.local_tensor()