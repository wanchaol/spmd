import torch
from distributed.tensor.api import Tensor
from distributed.tensor.ops.utils import register_impl


# def dist_detach(types, args=(), kwargs=None):
#     pass
    
@register_impl("aten.ones_like.default")
def dist_ones_like(types, args=(), kwargs=None):
    self_tensor = args[0]
    device_mesh = self_tensor.device_mesh

    new_local_tensor = torch.ones_like(self_tensor.local_tensor())
    return Tensor.from_local(new_local_tensor, device_mesh, self_tensor.placements)
    
# @register_impl("aten.expand.default")
# def dist_expand(types, args=(), kwargs=None):
#     self_tensor = args[0]
#     device_mesh = self_tensor.device_mesh

#     new_local_tensor = torch.ones_like(self_tensor.local_tensor())
#     return Tensor.from_local(new_local_tensor, device_mesh, self_tensor.placements)
    