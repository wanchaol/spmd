import torch
from torch.distributed.distributed_c10d import (
    ReduceOp
)
from distributed import (
    Tensor,
    Shard,
    Replicate,
    _Partial
)

def sharded_sum(types, args=(), kwargs=None):
    input = args[0]
    local_input = input.local_tensor()
    input_placement = input.placements[0]
    device_mesh = input.device_mesh

    local_sum = local_input.sum()

    if isinstance(input_placement, Shard) or isinstance(input_placement, _Partial):
        placements = [_Partial(ReduceOp.SUM)]
        # partial reduce
        partial_sum = Tensor.from_local(local_sum, placements, device_mesh)
        # all_reduce across device
        placements[0] = Replicate()
        return partial_sum.to_distributed(placements)
    elif isinstance(input_placement, Replicate):
        return Tensor.from_local(local_sum, placements=input.placements, device_mesh=device_mesh)
    else:
        raise RuntimeError("Not supported!")
