# These codes are copied from modelscope revision c58451baead80d83281f063d12fb377fad415257 
import os

import torch

from modelscope.utils.megatron_utils import init_megatron_util


def pre_compile_megatron_util():
    dummy_megatron_cfg = {
        'tensor_model_parallel_size': 1,
        'world_size': 1,
        'distributed_backend': 'nccl',
        'seed': 42,
    }
    os.environ['MASTER_PORT'] = '39501'
    init_megatron_util(dummy_megatron_cfg)


def pre_compile_all():
    if torch.cuda.is_available():  # extension require cuda.
        pre_compile_megatron_util()
        # pre compile pai-easycv
        from easycv.thirdparty.deformable_attention.functions import ms_deform_attn_func
        # extension for all platform.


if __name__ == '__main__':
    pre_compile_all()
