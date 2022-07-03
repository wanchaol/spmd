import torch
import torch.nn as nn
from .tensor import (
    DistributedTensor,
    DeviceMesh,
    PlacementSpec,
    Shard,
    Replicate,
    _Partial
)

torch.__future__.set_overwrite_module_params_on_conversion(True)

def distribute_tensor(tensor: torch.Tensor, spec: PlacementSpec):
    # distribute the tensor according to PlacementSpec
    placement_strategies = spec.placement_strategy
    device_mesh = spec.device_mesh
    assert len(placement_strategies) == 1, "Only support 1-d placement now"
    for idx, strategy in enumerate(placement_strategies):
        if isinstance(strategy, Shard):
            shard_dim = strategy.shard_dim
            assert shard_dim <= tensor.ndim, "Sharding dim {shard_dim} greater than tensor ndim {tensor.ndim}"
            # TODO: handle multi-dim device mesh and last shard
            num_chunks = device_mesh.size()
            assert tensor.size(shard_dim) % num_chunks == 0, "Only support chunk sharding evenly now"
            chunk_size = tensor.size(shard_dim) // num_chunks
            tensor_list = list(tensor.chunk(num_chunks, dim=shard_dim))
            scatter_shape = list(tensor.size())
            scatter_shape[shard_dim] = chunk_size 
            local_tensor = torch.empty(scatter_shape, dtype=tensor.dtype, device=tensor.device)
            device_mesh.scatter(
                local_tensor,
                tensor_list
            )
            dist_tensor = DistributedTensor.from_local(local_tensor, spec)
        elif isinstance(strategy, Replicate) or isinstance(strategy, _Partial):
            dist_tensor = DistributedTensor.from_local(tensor, spec)
        else:
            raise RuntimeError("Not supported!")

    return dist_tensor

def distribute_module(mod: nn.Module, spec: PlacementSpec):
    '''
    this function coverts all module parameters
    to distributed tensor parameters according to
    the PlacementSpec spcified.
    TODO: add a more flexible tagging, i.e. convert
    certain param to a certain spec, like a PlacementPlan
    '''
    def to_dist_tensor(t):
        if isinstance(t, nn.Parameter):
            print(f"!!! distributing tensor")
            return distribute_tensor(t.data, spec)
        else:
            return t

    mod._apply(to_dist_tensor)

    return mod