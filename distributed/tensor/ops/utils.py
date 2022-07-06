from distributed.tensor import (
    Tensor
)

def unwrap_single_strategy(e):
    if not isinstance(e, Tensor):
        return None
    assert len(e.placement_spec.placement_strategy) == 1, "more than one strategy"
    return e.placement_spec.placement_strategy[0]

def unwrap_local_tensor(e):
    if not isinstance(e, Tensor):
        return None
    return e.local_tensor()