import functools

import torch
from fairscale.nn.data_parallel.fully_sharded_data_parallel import FullyShardedDataParallel as FullyShardedDP
from fairscale.nn.wrap.auto_wrap import auto_wrap, enable_wrap, default_auto_wrap_policy


def default_initialize(model: torch.nn.Module,
                       device: torch.device,
                       fp16: bool = False,
                       flatten_parameters: bool = True,
                       reshard_after_forward: bool = True,
                       move_grads_to_cpu: bool = False,
                       move_params_to_cpu: bool = False):
    fsdp_params = dict(mixed_precision=fp16,
                       flatten_parameters=flatten_parameters,
                       reshard_after_forward=reshard_after_forward,
                       move_grads_to_cpu=move_grads_to_cpu,
                       move_params_to_cpu=move_params_to_cpu)
    model = FullyShardedDP(model, **fsdp_params)

    if not move_params_to_cpu:
        model = model.to(device)

    return model


def recursive_initialize(model: torch.nn.Module,
                         device: torch.device,
                         fp16: bool = False,
                         flatten_parameters: bool = True,
                         reshard_after_forward: bool = True,
                         move_grads_to_cpu: bool = False,
                         move_params_to_cpu: bool = False,
                         min_num_params: int = 1e8):
    wrap_policy = functools.partial(default_auto_wrap_policy,
                                    # module_is_root=True,  # TODO: Check if fairscale with current version supports the param.
                                    # force_leaf_modules=force_leaf_modules,
                                    min_num_params=min_num_params)
    fsdp_params = dict(mixed_precision=fp16,
                       flatten_parameters=flatten_parameters,
                       reshard_after_forward=reshard_after_forward,
                       move_grads_to_cpu=move_grads_to_cpu,
                       move_params_to_cpu=move_params_to_cpu)
    with enable_wrap(wrapper_cls=FullyShardedDP, auto_wrap_policy=wrap_policy, **fsdp_params):
        model = auto_wrap(model)
    model = FullyShardedDP(model, **fsdp_params)  # TODO: If fairscale supports ``model_is_root`` param, remove this line.
    assert isinstance(model, FullyShardedDP)

    if not move_params_to_cpu:
        model = model.to(device)

    return model
