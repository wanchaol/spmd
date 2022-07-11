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
    Tensor,
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
        device_mesh = DeviceMesh("cuda", list(range(self.world_size)))
        shard_spec = [Shard(0)]
        tensor_to_shard = torch.randn(12, 3, device="cuda")
        sharded_tensor = distribute_tensor(tensor_to_shard, device_mesh, shard_spec)
        self.assertEqual(sharded_tensor.size(), torch.Size([12, 3]))
        local_tensor = sharded_tensor.local_tensor()
        self.assertEqual(local_tensor.size(), torch.Size([3, 3]))

    @with_comms
    def test_distribute_module(self):
        device_mesh = DeviceMesh("cuda", list(range(self.world_size)))
        module_to_shard = MyModel(20, 20, device="cuda")
        shard_spec = [Shard(0)]
        sharded_module = distribute_module(module_to_shard, device_mesh, shard_spec)

        module_to_replicate = MyModel(20, 20, device="cuda").cuda()
        replica_spec = [Replicate()]
        replica_module = distribute_module(module_to_replicate, device_mesh, replica_spec)

class DDPWithDistTensorAPITest(DistTensorTestBase):
    @with_comms
    def test_full_replicated(self):
        device_mesh = DeviceMesh("cuda", list(range(self.world_size)))
        n_features = 10
        model = MyModel(n_features, n_features, device="cuda")
        # mark model as replication
        replica_spec = [Replicate()]
        replicated_model = distribute_module(model, device_mesh, replica_spec)

        input = torch.randn(10, n_features, device="cuda", requires_grad=True)
        # mark input as replicated
        replicated_input = Tensor.from_local(input, device_mesh, replica_spec)

        output = model(replicated_input)
        output.sum().backward()
        param_grad = list(model.parameters())[0].grad
        self.assertTrue(isinstance(param_grad, Tensor))
        self.assertTrue(isinstance(param_grad.placements[0], Replicate))

    @with_comms
    def test_ddp_dist_tensor(self):
        device_mesh = DeviceMesh("cuda", list(range(self.world_size)))
        n_features = 100
        model = MyModel(n_features, 1, device="cuda")
        # model = MyModel(20, 20, device="meta")
        # mark model as replication
        replica_spec = [Replicate()]
        replicated_model = distribute_module(model, device_mesh, replica_spec)

        shard0_spec = [Shard(0)]
        input = torch.randn(10, n_features, device="cuda")
        # mark input as shard on dim 0
        sharded_input = Tensor.from_local(input, device_mesh, shard0_spec)

        # run DDP like a normal model
        output = replicated_model(sharded_input)
        # output.sum().backward()


class DistTensorOpsTest(DistTensorTestBase):
    @with_comms
    def test_addmm(self):
        device_mesh = DeviceMesh("cuda", list(range(self.world_size)))
        shard_spec = [Shard(0)]
        replica_spec = [Replicate()]

        tensor_to_shard = torch.randn(12, 8, device="cuda")
        mat1 = distribute_tensor(tensor_to_shard, device_mesh, shard_spec)
        tensor_to_replicate = torch.randn(8, 4, device="cuda")
        mat2 = distribute_tensor(tensor_to_replicate, device_mesh, replica_spec)
        input_tensor = torch.randn(4, device="cuda")
        input = distribute_tensor(input_tensor, device_mesh, replica_spec)

        dist_res = torch.addmm(input, mat1, mat2)
        local_res = torch.addmm(input_tensor, tensor_to_shard, tensor_to_replicate)
        self.assertEqual(dist_res.redistribute(device_mesh, replica_spec).local_tensor(), local_res)

    @with_comms
    def test_mm(self):
        device_mesh = DeviceMesh("cuda", list(range(self.world_size)))
        shard_spec = [Shard(0)]
        replica_spec = [Replicate()]

        tensor_to_shard = torch.randn(12, 8, device="cuda", requires_grad=True)
        mat1 = distribute_tensor(tensor_to_shard, device_mesh, shard_spec)
        tensor_to_replicate = torch.randn(8, 4, device="cuda", requires_grad=True)
        mat2 = distribute_tensor(tensor_to_replicate, device_mesh, replica_spec)

        dist_res = torch.mm(mat1, mat2)
        local_res = torch.mm(tensor_to_shard, tensor_to_replicate)
        self.assertEqual(dist_res.redistribute(device_mesh, replica_spec).local_tensor(), local_res)
        # dist_res.sum().backward()

    @with_comms
    def test_t(self):
        device_mesh = DeviceMesh("cuda", list(range(self.world_size)))
        shard_spec = [Shard(0)]
        replica_spec = [Replicate()]

        tensor_to_transpose = torch.randn(12, 8, device="cuda", requires_grad=True)
        mat = distribute_tensor(tensor_to_transpose, device_mesh, shard_spec)
        tranposed_mat = mat.t()
        self.assertEqual(tranposed_mat.size(), torch.Size([8, 12]))
        self.assertEqual(tranposed_mat.placements, [Shard(1)])
        tranposed_mat2 = tranposed_mat.t()
        self.assertEqual(tranposed_mat2.size(), torch.Size([12, 8]))
        self.assertEqual(tranposed_mat2.placements, shard_spec)


if __name__ == '__main__':
    run_tests()
