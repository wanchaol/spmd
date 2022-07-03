import torch
from torch.distributed.distributed_c10d import (
    # all_gather,
    _all_gather_base,
    barrier,
    broadcast,
    scatter,
    get_rank,
    ReduceOp,
    ProcessGroup,
    _get_default_group
)

class DeviceMesh(object):
    '''
    Device Mesh object
    By default describes the device ids, layout and serves
    as a proxy for communication among the device lists.

    We use the default ProcessGroup in this DeviceMesh class
    to implement proper communications. Note that we also
    add collective wrappers in this class. This is used to
    decouple detailed communication backend with the underlying
    DistributedTensor implementation.
    '''
    mesh: torch.Tensor
    # _world_pg: ProcessGroup

    def __init__(self, mesh):
        self.mesh = mesh
        # self._world_pg = _get_default_group()

        # TODO: support multi-dimensional device mesh
        assert self.mesh.ndim == 1, "Only support 1-d device mesh for now"

    def __repr__(self):
        return f"DeviceMesh:({self.mesh})"

    def size(self, dim=0):
        return self.mesh.size(dim)

    def get_rank(self):
        return get_rank()

    def scatter(self, tensor, scatter_list=None, src=0):
        current_rank = get_rank()
        return scatter(tensor,
                       scatter_list=scatter_list if current_rank == src else None,
                       src=src)

    def broadcast(self, tensor, src=0):
        return broadcast(tensor, src=src)

    def barrier(self):
        barrier()

    # def all_gather(self, tensor_list, tensor):
    #     return all_gather(tensor_list, tensor)

    def all_gather_base(self, output_tensor, tensor):
        return _all_gather_base(output_tensor, tensor)

    def all_reduce(self, tensor, op=ReduceOp.SUM):
        return all_reduce(tensor, op=op)