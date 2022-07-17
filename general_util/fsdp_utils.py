import functools
from functools import partial
from typing import Dict, Set, Type

import torch
from fairscale.nn.data_parallel.fully_sharded_data_parallel import FullyShardedDataParallel as FullyShardedDP
from fairscale.nn.wrap.auto_wrap import auto_wrap, enable_wrap, default_auto_wrap_policy
from torch import nn
from transformers.models.t5.modeling_t5 import T5Block

from general_util.logger import get_child_logger

logger = get_child_logger("FSDPUtils")


def transformer_auto_wrap_policy(
        module: nn.Module,
        recurse: bool,
        unwrapped_params: int,
        module_is_root: bool,
        transformer_layer_cls: Set[Type[nn.Module]],
) -> bool:
    """
    A convenient auto wrap policy for transformer models. If the submodule
    is an instance of transformer_layer_cls, the submodule will be wrapped
    as a FSDP unit. Otherwise, all the other remainder submodules are wrapped
    by the outermost FSDP unit. Right now, FSDP requires submodules that share
    weights to be wrapped in the same FSDP unit, this auto wrap policy can
    conviniently wrap the shared embeddings into the same FSDP unit for transformer
    models. In the near future, FSDP will support submodules that share weights
    to be wrapped in the separated FSDP units.

    Return if a module should be wrapped during FSDP auto wrapping.

    The first three parameters are required by :func:`_recursive_wrap`.


    Args:
       module (nn.Module):
           The module to be considered in this decision.
       recurse (bool):
           Indicate if this is called to make a decision on whether we
           should recurse down a subgraph of the module structure.
           If False, it means this function is called to make a decision
           on whether we should wrap the said module.
       unwrapped_params (int):
           The number of parameters yet to be wrapped in this module.

       transformer_layer_cls (int):
           Submodules with one of the `transformer_layer_cls` names
           will be wrapped as seperated FSDP units
    """
    if recurse:
        # always recurse
        return True
    else:
        # if not recursing, decide whether we should wrap for the leaf node or reminder
        return isinstance(module, tuple(transformer_layer_cls))


def default_initialize(model: torch.nn.Module,
                       device: torch.device,
                       fp16: bool = False,
                       flatten_parameters: bool = True,
                       disable_reshard_on_root: bool = True,
                       reshard_after_forward: bool = True,
                       move_grads_to_cpu: bool = False,
                       move_params_to_cpu: bool = False):
    fsdp_params = dict(mixed_precision=fp16,
                       flatten_parameters=flatten_parameters,
                       disable_reshard_on_root=disable_reshard_on_root,
                       reshard_after_forward=reshard_after_forward,
                       move_grads_to_cpu=move_grads_to_cpu,
                       move_params_to_cpu=move_params_to_cpu)

    # Better speed

    logger.info(fsdp_params)

    model = FullyShardedDP(model, **fsdp_params)

    if not move_params_to_cpu:
        model = model.to(device)

    return model


def transformer_init(model,
                     device,
                     fp16: bool = False,
                     flatten_parameters: bool = True,
                     disable_reshard_on_root: bool = True,
                     reshard_after_forward: bool = True,
                     move_grads_to_cpu: bool = False,
                     move_params_to_cpu: bool = False, ):
    fsdp_params = dict(mixed_precision=fp16,
                       flatten_parameters=flatten_parameters,
                       disable_reshard_on_root=disable_reshard_on_root,
                       reshard_after_forward=reshard_after_forward,
                       move_grads_to_cpu=move_grads_to_cpu,
                       move_params_to_cpu=move_params_to_cpu)

    wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={T5Block})

    with enable_wrap(wrapper_cls=FullyShardedDP, auto_wrap_policy=wrap_policy, **fsdp_params):
        model = auto_wrap(model)
    model = FullyShardedDP(model, **fsdp_params)

    logger.info(model)

    assert isinstance(model, FullyShardedDP)

    if not move_params_to_cpu:
        model = model.to(device)

    return model


def default_initialize_w_mp(model: torch.nn.Module,
                            device: torch.device,
                            fp16: bool = False,
                            flatten_parameters: bool = True,
                            disable_reshard_on_root: bool = True,
                            reshard_after_forward: bool = True,
                            move_grads_to_cpu: bool = False,
                            move_params_to_cpu: bool = False,
                            n_gpu: int = 1,
                            device_map: Dict = None):
    fsdp_params = dict(mixed_precision=fp16,
                       flatten_parameters=flatten_parameters,
                       disable_reshard_on_root=disable_reshard_on_root,
                       reshard_after_forward=reshard_after_forward,
                       move_grads_to_cpu=move_grads_to_cpu,
                       move_params_to_cpu=move_params_to_cpu)

    # Better speed

    logger.info(fsdp_params)

    model = FullyShardedDP(model, **fsdp_params)

    if not move_params_to_cpu:
        # model = model.to(device)
        if n_gpu == 1:
            model.to(device)
        else:
            # For model parallel (of mT5)
            model.parallelize(device_map)

    return model


def recursive_initialize(model: torch.nn.Module,
                         device: torch.device,
                         fp16: bool = False,
                         flatten_parameters: bool = True,
                         disable_reshard_on_root: bool = True,
                         reshard_after_forward: bool = True,
                         move_grads_to_cpu: bool = False,
                         move_params_to_cpu: bool = False,
                         min_num_params: int = 1e8):
    # Better memory?
    wrap_policy = functools.partial(default_auto_wrap_policy,
                                    module_is_root=True,
                                    # force_leaf_modules=force_leaf_modules,
                                    min_num_params=min_num_params)
    fsdp_params = dict(mixed_precision=fp16,
                       flatten_parameters=flatten_parameters,
                       disable_reshard_on_root=disable_reshard_on_root,
                       reshard_after_forward=reshard_after_forward,
                       move_grads_to_cpu=move_grads_to_cpu,
                       move_params_to_cpu=move_params_to_cpu)
    with enable_wrap(wrapper_cls=FullyShardedDP, auto_wrap_policy=wrap_policy, **fsdp_params):
        model = auto_wrap(model)
    model = FullyShardedDP(model, **fsdp_params)

    logger.info(model)

    assert isinstance(model, FullyShardedDP)

    if not move_params_to_cpu:
        model = model.to(device)

    return model
