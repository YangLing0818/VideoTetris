# These codes are copied from modelscope revision c58451baead80d83281f063d12fb377fad415257 
# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from transformers.models.llama import (LlamaConfig, LlamaTokenizer,
                                       LlamaTokenizerFast)

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .backbone import LlamaModel
    from .text_generation import LlamaForTextGeneration
else:
    _import_structure = {
        'backbone': ['LlamaModel'],
        'text_generation': ['LlamaForTextGeneration'],
    }
    _extra_objects = {
        'LlamaConfig': LlamaConfig,
        'LlamaTokenizer': LlamaTokenizer,
        'LlamaTokenizerFast': LlamaTokenizerFast,
    }
    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects=_extra_objects,
    )
