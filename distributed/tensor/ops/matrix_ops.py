# implement matrix related ops for distributed tensor
import torch.utils._pytree as pytree
from torch.distributed.distributed_c10d import (
    ReduceOp
)
from distributed.tensor.api import Tensor
from distributed.tensor.placement_types import (
    Shard,
    Replicate,
    _Partial
)
from distributed.tensor.ops.utils import (
    unwrap_local_tensor,
    unwrap_single_placement,
    is_shard_on_dim,
    register_impl
)


@register_impl("aten.addmm.default")
def dist_addmm(types, args=(), kwargs=None):
    # dist addmm:
    # input:shard(0)    mat1: shard(0),  mat2: replicate
    # input:shard(1)    mat1: replicate, mat2: shard(1)
    # input:replicate   mat1: shard(0),  mat2: replicate
    # input:replicate   mat1: replicate, mat2: shard(1)
    # input:replicate   mat1: shard(0),  mat2: shard(1)
    input, mat1, mat2 = args
    local_input, local_mat1, local_mat2 = pytree.tree_map(unwrap_local_tensor, args)
    input_placement, mat1_placement, mat2_placement = pytree.tree_map(unwrap_single_placement, args)
    beta = kwargs.get("beta", 1)
    alpha = kwargs.get("alpha", 1)
    device_mesh = mat1.device_mesh
    world_size = device_mesh.size()
    current_rank = device_mesh.get_rank()

    assert isinstance(input_placement, Replicate), "only support replication now"
    
    # only implemented combo with no comm for now
    # TODO: implement all combinations
    if isinstance(mat1_placement, Shard) and isinstance(mat2_placement, Replicate):
        mat1_shard_dim = mat1_placement.dim
        chunk_size = mat1.size(0) // world_size
        assert mat1_shard_dim == 0, "shard dim should be 0!"
        local_res = local_input.addmm(local_mat1, local_mat2, beta=beta, alpha=alpha)
        return Tensor.from_local(local_res, device_mesh, mat1.placements)
    elif isinstance(mat1_placement, Replicate) and isinstance(mat2_placement, Shard):
        mat2_shard_dim = mat2_placement.dim
        assert mat2_shard_dim == 1, "shard dim should be 1!"
        chunk_size = mat1.size(1) // world_size
        local_res = local_input.addmm(local_mat1, local_mat2, beta=beta, alpha=alpha)
        return Tensor.from_local(local_res, device_mesh, mat2.placements)
    elif isinstance(mat1_placement, Replicate) and isinstance(mat2_placement, Replicate):
        local_res = local_input.addmm(local_mat1, local_mat2, beta=beta, alpha=alpha)
        return Tensor.from_local(local_res, device_mesh, mat1.placement, run_check=False)
    else:
        raise RuntimeError(f"addmm operator supported for inputs: {mat1}, {mat2}")

@register_impl("aten.mm.default")
def dist_mm(types, args=(), kwargs=None):
    # dist mm:
    # mat1: shard(0),  mat2: replicate
    # mat1: replicate, mat2: shard(1)
    # mat1: shard(1),  mat2: shard(0)
    # mat1: shard(0),  mat2: shard(1)
    mat1, mat2 = args
    local_mat1, local_mat2 = pytree.tree_map(unwrap_local_tensor, args)
    mat1_placement, mat2_placement = pytree.tree_map(unwrap_single_placement, args)
    device_mesh = mat1.device_mesh

    # print(f"?????!!! sharded mm mat1 placement {mat1_placement}, mat2 placement: {mat2_placement}")

    # only implemented the first 3
    # TODO: implement all combinations
    if is_shard_on_dim(mat1_placement, 0) and isinstance(mat2_placement, Replicate):
        local_res = local_mat1.mm(local_mat2)
        return Tensor.from_local(local_res, device_mesh, mat1.placements)
    elif isinstance(mat1_placement, Replicate) and is_shard_on_dim(mat2_placement, 1):
        local_res = local_mat1.mm(local_mat2)
        return Tensor.from_local(local_res, device_mesh, mat2.placements)
    elif is_shard_on_dim(mat1_placement, 1) and is_shard_on_dim(mat2_placement, 0):
        local_res = local_mat1.mm(local_mat2)
        placements = [_Partial(ReduceOp.SUM)]
        partial_sum = Tensor.from_local(local_res, device_mesh, placements)
        # all reduce across ranks
        placements[0] = [Replicate()]
        return partial_sum.redistribute(device_mesh, placements)
    else:
        raise RuntimeError(f"mm operator supported for inputs: {mat1}, {mat2}")


@register_impl("aten.t.default")
def dist_t(types, args=(), kwargs=None):
    # transpose with sharding
    mat = args[0]
    local_mat = pytree.tree_map(unwrap_local_tensor, mat)
    assert local_mat.ndim == 2
    mat_placement = pytree.tree_map(unwrap_single_placement, mat)
    transposed_local_mat = local_mat.t()
    device_mesh = mat.device_mesh

    new_shard_dim = 1 if is_shard_on_dim(mat_placement, 0) else 0
    return Tensor.from_local(transposed_local_mat, device_mesh, [Shard(new_shard_dim)])
