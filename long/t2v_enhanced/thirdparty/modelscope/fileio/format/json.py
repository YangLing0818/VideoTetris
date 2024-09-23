# These codes are copied from modelscope revision c58451baead80d83281f063d12fb377fad415257 
# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np

from . import jsonplus
from .base import FormatHandler


def set_default(obj):
    """Set default json values for non-serializable values.

    It helps convert ``set``, ``range`` and ``np.ndarray`` data types to list.
    It also converts ``np.generic`` (including ``np.int32``, ``np.float32``,
    etc.) into plain numbers of plain python built-in types.
    """
    if isinstance(obj, (set, range)):
        return list(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f'{type(obj)} is unsupported for json dump')


class JsonHandler(FormatHandler):
    """Use jsonplus, serialization of Python types to JSON that "just works"."""

    def load(self, file):
        return jsonplus.loads(file.read())

    def dump(self, obj, file, **kwargs):
        file.write(self.dumps(obj, **kwargs))

    def dumps(self, obj, **kwargs):
        return jsonplus.dumps(obj, **kwargs)
