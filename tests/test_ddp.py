import torch
import torch.nn as nn
from torch.testing._internal.common_utils import (
    run_tests
)
from .utils import DistTensorTestBase, with_comms
from distributed import (
    distribute_tensor,
    distribute_module,
    DeviceMesh,
    DistributedTensor,
    PlacementSpec,
    Shard,
    Replicate
)

class MyModel(nn.Module):
    def __init__(self, n_features, n_layers, device):
        super().__init__()
        self.seq = nn.Sequential(*[nn.Linear(n_features, n_features, device=device) for _ in range(n_layers)])

    def forward(self, x):
        return self.seq(x)

    def reset_parameters(self):
        for m in self.seq:
            m.reset_parameters()

class DistTensorAPITest(DistTensorTestBase):
    @with_comms
    def test_distribute_tensor(self):
        device_mesh = DeviceMesh(torch.arange(self.world_size))
        shard_spec = PlacementSpec(device_mesh, placement_strategy=[Shard(0)])
        tensor_to_shard = torch.randn(12, 3, device="cuda")
        sharded_tensor = distribute_tensor(tensor_to_shard, shard_spec)
        self.assertEqual(sharded_tensor.size(), torch.Size([12, 3]))
        local_tensor = sharded_tensor.local_tensor()
        self.assertEqual(local_tensor.size(), torch.Size([3, 3]))

    @with_comms
    def test_distribute_module(self):
        device_mesh = DeviceMesh(torch.arange(self.world_size))
        module_to_shard = MyModel(20, 20, device="cuda")
        shard_spec = PlacementSpec(device_mesh, placement_strategy=[Shard(0)])
        sharded_module = distribute_module(module_to_shard, shard_spec)

        module_to_replicate = MyModel(20, 20, device="cuda").cuda()
        replica_spec = PlacementSpec(device_mesh, placement_strategy=[Replicate()])
        replica_module = distribute_module(module_to_replicate, replica_spec)

class DDPWithDistTensorAPITest(DistTensorTestBase):
    @with_comms
    def test_ddp_dist_tensor(self):
        device_mesh = DeviceMesh(torch.arange(self.world_size))
        n_features = 100
        model = MyModel(n_features, 1, device="cuda")
        # model = MyModel(20, 20, device="meta")
        # mark model as replication
        replica_spec = PlacementSpec(device_mesh, placement_strategy=[Replicate()])
        replicated_model = distribute_module(model, replica_spec)

        shard0_spec = PlacementSpec(device_mesh, placement_strategy=[Shard(0)])
        input = torch.randn(10, n_features, device="cuda")
        # mark input as shard on dim 0
        sharded_input = DistributedTensor.from_local(input, shard0_spec)

        # run DDP like a normal model
        output = replicated_model(sharded_input)


class DistTensorOpsTest(DistTensorTestBase):
    @with_comms
    def test_addmm(self):
        device_mesh = DeviceMesh(torch.arange(self.world_size))
        shard_spec = PlacementSpec(device_mesh, placement_strategy=[Shard(0)])
        replica_spec = PlacementSpec(device_mesh, placement_strategy=[Replicate()])

        tensor_to_shard = torch.randn(12, 8, device="cuda")
        mat1 = distribute_tensor(tensor_to_shard, shard_spec)
        tensor_to_replicate = torch.randn(8, 4, device="cuda")
        mat2 = distribute_tensor(tensor_to_replicate, replica_spec)
        input_tensor = torch.randn(4, device="cuda")
        input = distribute_tensor(input_tensor, replica_spec)

        dist_res = torch.addmm(input, mat1, mat2)
        local_res = torch.addmm(input_tensor, tensor_to_shard, tensor_to_replicate)
        self.assertEqual(dist_res.to_global(), local_res)


if __name__ == '__main__':
    run_tests()