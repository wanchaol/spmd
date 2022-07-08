import torch

from torch.distributed.distributed_c10d import (
    ReduceOp
)

from torch.testing._internal.common_utils import (
    run_tests
)
from .utils import DistTensorTestBase, with_comms
from distributed.tensor import (
    DeviceMesh,
    Tensor,
    Replicate,
    Shard,
    _Partial
)

class DistTensorTest(DistTensorTestBase):
    # @with_comms
    # def test_tensor_constructor(self):
    #     import distributed.tensor as dist_tensor
    #     shard_spec = PlacementSpec(device_mesh, strategies=[Shard(0)])
    #     empty_tensor = dist_tensor.empty((12, 10), placement_spec=shard_spec)
    #     zero_tensor = dist_tensor.zeros((12, 10), placement_spec=shard_spec)
    #     one_tensor = dist_tensor.ones((12, 10), placement_spec=shard_spec)

    #     zero_cuda_tensor = dist_tensor.zeros((12, 10), device="cuda", placement_spec=shard_spec)

    #     dist_tensor.empty_like(empty_tensor)
    #     dist_tensor.zero_like(empty_tensor)
    #     dist_tensor.one_like(empty_tensor)

    @with_comms
    def test_tensor_from_local(self):
        device_mesh = DeviceMesh(*range(self.world_size))
        shard_spec = [Shard(0)]
        local_tensor = torch.randn(3, 3, device="cuda")
        sharded_tensor = Tensor.from_local(local_tensor, device_mesh, shard_spec)
        self.assertEqual(sharded_tensor.size(), torch.Size([12, 3]))

        replica_spec = [Replicate()]
        ddp_tensor = Tensor.from_local(local_tensor, device_mesh, replica_spec)
        self.assertEqual(ddp_tensor.size(), local_tensor.size())

        partial_spec = [_Partial(ReduceOp.SUM)]
        partial_tensor = Tensor.from_local(local_tensor, device_mesh, partial_spec)
        self.assertEqual(partial_tensor.size(), local_tensor.size())

    @with_comms
    def test_tensor_redistribute(self):
        # test sharding a tensor, then get the global tensor
        device_mesh = DeviceMesh(*range(self.world_size))
        shard_dim = 0
        shard_spec = [Shard(shard_dim)]
        replica_spec = [Replicate()]
        expected_tensor = torch.randn(12, 3, device="cuda")
        chunked_list = expected_tensor.chunk(self.world_size, shard_dim)
        # make local tensor as the element of the corresponding chunked list
        local_tensor = chunked_list[self.rank]
        sharded_tensor = Tensor.from_local(local_tensor, device_mesh, shard_spec)
        global_sharded_tensor = sharded_tensor.redistribute(device_mesh, replica_spec).local_tensor()
        self.assertEqual(global_sharded_tensor.size(), torch.Size([12, 3]))
        self.assertEqual(expected_tensor, global_sharded_tensor)

        # test replicating a tensor, then get self
        ddp_tensor = Tensor.from_local(local_tensor, device_mesh, replica_spec)
        global_ddp_tensor = ddp_tensor.redistribute(device_mesh, replica_spec)
        self.assertEqual(ddp_tensor.size(), local_tensor.size())

        # test creating a partial tensor, then get the global tensor
        # note that the global tensor should get all reduced
        partial_spec = [_Partial(ReduceOp.SUM)]
        partial_tensor = Tensor.from_local(local_tensor, device_mesh, partial_spec)
        global_partial_tensor = partial_tensor.redistribute(device_mesh, replica_spec)
        self.assertEqual(partial_tensor.size(), local_tensor.size())

    @with_comms
    def test_placement_spec_read_only_after_set(self):
        device_mesh = DeviceMesh(*range(self.world_size))
        shard_spec = [Shard(0)]
        local_tensor = torch.randn(3, 3, device="cuda")
        sharded_tensor = Tensor.from_local(local_tensor, device_mesh, shard_spec)

        # modify shard_spec, and dist_tensor's spec should not be changed
        shard_spec[0]=Replicate()
        self.assertTrue(sharded_tensor.placements is not shard_spec)
        self.assertNotEqual(sharded_tensor.placements, shard_spec)

    @with_comms
    def test_tensor_properties(self):
        device_mesh = DeviceMesh(*range(self.world_size))
        shard_spec = [Shard(0)]
        local_tensor = torch.randn(3, 3, device="cuda")
        sharded_tensor = Tensor.from_local(local_tensor, device_mesh, shard_spec)
        print(sharded_tensor.device)


if __name__ == '__main__':
    run_tests()

