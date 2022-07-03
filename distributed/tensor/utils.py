import functools
# from .api import DistributedTensor

def all_equal(xs):
  xs = list(xs)
  if not xs:
    return True
  return xs[1:] == xs[:-1]

# def register_op(func, op):
#     assert func is not None
#     @functools.wraps(func)
#     def wrapper(types, args, kwargs):
#         return func(types, args, kwargs)
#     # update ops table
#     DistributedTensor._dist_tensor_dispatch_ops[func]
#     return wrapper

