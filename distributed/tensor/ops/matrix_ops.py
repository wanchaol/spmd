# implement matrix related ops for distributed tensor
import torch.utils._pytree as pytree
from distributed.tensor import (
    Tensor,
    Shard,
    Replicate
)
from .utils import unwrap_local_tensor, unwrap_single_placement
# from distributed.tensor.utils import register_op

# sharded addmm:
# input:shard(0)    mat1: shard(0),  mat2: replicate
# input:shard(1)    mat1: replicate, mat2: shard(1)
# input:replicate   mat1: shard(0),  mat2: replicate
# input:replicate   mat1: replicate, mat2: shard(1)
# input:replicate   mat1: shard(0),  mat2: shard(1)
def sharded_addmm(types, args=(), kwargs=None):
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
        return Tensor.from_local(local_res, mat1.placements, device_mesh)
    elif isinstance(mat1_placement, Replicate) and isinstance(mat2_placement, Shard):
        mat2_shard_dim = mat2_placement.dim
        assert mat2_shard_dim == 1, "shard dim should be 1!"
        chunk_size = mat1.size(1) // world_size
        local_res = local_input.addmm(local_mat1, local_mat2, beta=beta, alpha=alpha)
        return Tensor.from_local(local_res, mat2.placements, device_mesh)
    elif isinstance(mat1_placement, Replicate) and isinstance(mat2_placement, Replicate):
        local_res = local_input.addmm(local_mat1, local_mat2, beta=beta, alpha=alpha)
        return Tensor.from_local(local_res, mat1.placement, device_mesh, run_check=False)
    else:
        raise RuntimeError(f"addmm operator supported for inputs: {mat1}, {mat2}")