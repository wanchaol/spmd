import copy
import torch
import torch.utils._pytree as pytree
from typing import Dict, Callable
from .placement_spec import (
    PlacementSpec,
    Shard,
    Replicate,
    _Partial
)
from .utils import all_equal

class DistributedTensor(torch.Tensor):
    _local_tensor: torch.Tensor
    _placement_spec: PlacementSpec

    __slots__ = ['_local_tensor', '_placement_spec']

    # class attribute that handles ops, all handled
    # ops should appear in this table
    _dist_tensor_dispatch_ops: Dict[Callable, Callable] = {}

    # context = contextlib.nullcontext

    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(cls, placement_spec: PlacementSpec, size: torch.Size, **kwargs):
        # new method instruct wrapper tensor and add placement spec
        # it does not do actual distribution, __init__ should do it instead.
        # TODO: implement __init__ for tensor constructors
        assert isinstance(placement_spec, PlacementSpec)
        # sizes = _flatten_tensor_size(size)
        dtype = kwargs['dtype']
        layout = kwargs['layout']
        requires_grad = kwargs['requires_grad']
        
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            size,
            dtype=dtype,
            layout=layout,
            requires_grad=requires_grad
        )
        # deepcopy and set spec, data should be handled
        # by __init__ or from_local instead.
        r._placement_spec = copy.deepcopy(placement_spec)
        return r

    def __repr__(self):
        # TODO: consider all_gather the local tensors for better debugging
        return f"DistributedTensor({self._local_tensor}, {self._placement_spec})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap_spec(e):
            # if this tensor is not Distributed, then return none. We will reinterpret it as replicated
            if not isinstance(e, DistributedTensor):
                return None
            return e.placement_spec
        
        def unwrap(e):
            if not isinstance(e, DistributedTensor):
                return None
            return e._local_tensor

        def wrap(e, spec):
            if isinstance(e, DistributedTensor):
                return e
            return DistributedTensor.from_local(e, spec, run_check=False)

        args_spec = pytree.tree_map(unwrap_spec, args)
        # assert all_equal(spec.device_mesh for spec in args_spec), "can't compuate across different meshes"
        for spec in args_spec:
            assert spec.device_mesh.mesh.ndim == 1, "Only 1-D mesh supported now"

        # with cls.context():
        #     rs = tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))
        from .ops import sharded_addmm
        if str(func) == "aten.addmm.default":
            return sharded_addmm(types, args, kwargs)
        else:
            # default to local tensor ops, this is wrong
            # but we use it now to enable more tensor property access 
            rs = func(*pytree.tree_map(unwrap, args), **pytree.tree_map(unwrap, kwargs))
            rs = wrap(rs, args_spec[0])
            return rs


    @classmethod
    def from_local(cls, local_tensor, placement_spec, run_check=True):
        # if same shape/dtype, no need to run_check, if not, must allgather
        # the metadatas to check the size/dtype across ranks
        # There should be no data communication unless there's replication
        # strategy, where we broadcast the replication from rank 0
        tensor_shape = list(local_tensor.size())
        placement_strategies = placement_spec.placement_strategy
        device_mesh = placement_spec.device_mesh
        assert len(placement_strategies) == 1, "only support 1-d placement now"
        for idx, strategy in enumerate(placement_strategies):
            # device_list = device_mesh.mesh[idx]
            if isinstance(strategy, Shard):
                shard_dim = strategy.shard_dim
                # recover tensor shape on the shard dim
                tensor_shape[shard_dim] = tensor_shape[shard_dim] * device_mesh.mesh.size(idx)
            elif isinstance(strategy, Replicate):
                # broadcast rank 0 tensor to all ranks
                local_tensor = local_tensor.contiguous()
                device_mesh.broadcast(local_tensor, 0)
            elif isinstance(strategy, _Partial):
                # we don't need to do anything to Partial case
                pass
            else:
                raise RuntimeError(f"strategy type {type(strategy)} not supported!")
        
        if run_check:
            # TODO: by default check tensor metas across rank
            pass

        dist_tensor = cls(
            placement_spec,
            torch.Size(tensor_shape),
            dtype=local_tensor.dtype,
            layout=local_tensor.layout,
            requires_grad=local_tensor.requires_grad
        )
        dist_tensor._local_tensor = local_tensor
        return dist_tensor


    def to_global(self) -> torch.Tensor:
        # TODO: handle last shard uneven with padding
        # right now we assume all local shard equal size
        # NOTE: this call materializes the distributed tensor
        # on all rank by doing all_gather, it will increase
        # memory usage and potentially OOM if the global tensor
        # is too big to fit into CPU/CUDA memory.
        placement_strategies = self._placement_spec.placement_strategy
        device_mesh = self._placement_spec.device_mesh
        assert len(placement_strategies) == 1, "Only support 1-d placement for now"
        for idx, strategy in enumerate(placement_strategies):
            # device_list = device_mesh.mesh[idx]
            if isinstance(strategy, Shard):
                # for shard, all_gather all shards and return the global tensor
                # shard_dim = strategy.shard_dim
                # num_shards = device_mesh.size()
                global_tensor = torch.empty(
                    self.size(),
                    device=self._local_tensor.device,
                    dtype=self.dtype
                )
                # NOTE: all_gather_base only works well when tensor
                # sharded on a sequential list of devices
                device_mesh.all_gather_base(global_tensor, self._local_tensor)
                return global_tensor
            elif isinstance(strategy, Replicate):
                # for replicate, just return its local tensor
                return self.local_tensor
            elif isinstance(strategy, _Partial):
                # for partial, trigger all_reduce and return the reduced result
                device_mesh.all_reduce(self._local_tensor, strategy.reduce_op)
                # update the PlacementSpec after all_reduce
                self._placement_spec[idx] = Replicate()
                return self.local_tensor
            else:
                raise RuntimeError(f"strategy type {type(strategy)} not supported!")

    def to_distributed(self, spec: PlacementSpec) -> "DistributedTensor":
        # This API perform necessary transformations and get
        # a new DistributedTensor with the new spec. i.e. for
        # sharding it's a reshard behavior.
        # TODO: implement to_distributed
        pass

    def local_tensor(self) -> torch.Tensor:
        return self._local_tensor

    @property
    def placement_spec(self):
        # placement_spec should be a read only propety
        # to disallow caller modification on it
        # caller who want a different PlacementSpec
        # should call to_distributed instead.
        return self._placement_spec
