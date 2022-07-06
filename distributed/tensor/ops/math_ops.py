import torch
from torch.distributed.distributed_c10d import (
    ReduceOp
)
from distributed import (
    PlacementSpec,
    Tensor,
    Shard,
    Replicate,
    _Partial
)
from .utils import unwrap_local_tensor, unwrap_single_strategy


def sharded_sum(types, args=(), kwargs=None):
    input = args[0]
    local_input = unwrap_local_tensor(input)
    input_strategy = unwrap_single_strategy(input)
    device_mesh = input.placement_spec.device_mesh

    local_sum = local_input.sum()

    if isinstance(input_strategy, Shard) or isinstance(input_strategy, _Partial):
        placement_spec = PlacementSpec(device_mesh, strategies=[_Partial(ReduceOp.SUM)])
        # partial reduce
        partial_sum = Tensor.from_local(local_sum, placement_spec)
        # all_reduce across device
        placement_spec.strategies[0] = Replicate()
        return partial_sum.to_distributed(placement_spec)
    elif isinstance(input_strategy, Replicate):
        return Tensor.from_local(local_sum, placement_spec=input.placement_spec)
    else:
        raise RuntimeError("Not supported!")
