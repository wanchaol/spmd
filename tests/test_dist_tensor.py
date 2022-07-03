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
    DistributedTensor,
    PlacementSpec,
    Replicate,
    Shard,
    _Partial
)

class DistTensorTest(DistTensorTestBase):

    @with_comms
    def test_tensor_from_local(self):
        device_mesh = DeviceMesh(torch.arange(self.world_size))
        shard_spec = PlacementSpec(device_mesh, placement_strategy=[Shard(0)])
        local_tensor = torch.randn(3, 3, device="cuda")
        sharded_tensor = DistributedTensor.from_local(local_tensor, shard_spec)
        self.assertEqual(sharded_tensor.size(), torch.Size([12, 3]))

        replica_spec = PlacementSpec(device_mesh, placement_strategy=[Replicate()])
        ddp_tensor = DistributedTensor.from_local(local_tensor, replica_spec)
        self.assertEqual(ddp_tensor.size(), local_tensor.size())

        partial_spec = PlacementSpec(device_mesh, placement_strategy=[_Partial(ReduceOp.SUM)])
        partial_tensor = DistributedTensor.from_local(local_tensor, partial_spec)
        self.assertEqual(partial_tensor.size(), local_tensor.size())

    @with_comms
    def test_tensor_to_global(self):
        # test sharding a tensor, then get the global tensor
        device_mesh = DeviceMesh(torch.arange(self.world_size))
        shard_dim = 0
        shard_spec = PlacementSpec(device_mesh, placement_strategy=[Shard(shard_dim)])
        expected_tensor = torch.randn(12, 3, device="cuda")
        chunked_list = expected_tensor.chunk(self.world_size, shard_dim)
        # make local tensor as the element of the corresponding chunked list
        local_tensor = chunked_list[self.rank]
        sharded_tensor = DistributedTensor.from_local(local_tensor, shard_spec)
        global_sharded_tensor = sharded_tensor.to_global()
        self.assertEqual(global_sharded_tensor.size(), torch.Size([12, 3]))
        self.assertEqual(expected_tensor, global_sharded_tensor)

        # test replicating a tensor, then get the global tensor
        replica_spec = PlacementSpec(device_mesh, placement_strategy=[Replicate()])
        ddp_tensor = DistributedTensor.from_local(local_tensor, replica_spec)
        global_ddp_tensor = ddp_tensor.to_global()
        self.assertEqual(ddp_tensor.size(), local_tensor.size())

        # test creating a partial tensor, then get the global tensor
        # note that the global tensor should get all reduced
        partial_spec = PlacementSpec(device_mesh, placement_strategy=[_Partial(ReduceOp.SUM)])
        partial_tensor = DistributedTensor.from_local(local_tensor, partial_spec)
        global_partial_tensor = partial_tensor.to_global()
        self.assertEqual(partial_tensor.size(), local_tensor.size())

    @with_comms
    def test_placement_spec_read_only_after_set(self):
        device_mesh = DeviceMesh(torch.arange(self.world_size))
        shard_spec = PlacementSpec(device_mesh, placement_strategy=[Shard(0)])
        local_tensor = torch.randn(3, 3, device="cuda")
        sharded_tensor = DistributedTensor.from_local(local_tensor, shard_spec)

        # modify shard_spec, and dist_tensor's spec should not be changed
        shard_spec.placement_strategy=[Replicate()]
        self.assertTrue(sharded_tensor.placement_spec is not shard_spec)
        self.assertNotEqual(sharded_tensor.placement_spec, shard_spec)


if __name__ == '__main__':
    run_tests()

