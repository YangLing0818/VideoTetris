# These codes are copied from modelscope revision c58451baead80d83281f063d12fb377fad415257 
# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .cartoon_translation_exporter import CartoonTranslationExporter
    from .object_detection_damoyolo_exporter import ObjectDetectionDamoyoloExporter
    from .face_detection_scrfd_exporter import FaceDetectionSCRFDExporter
else:
    _import_structure = {
        'cartoon_translation_exporter': ['CartoonTranslationExporter'],
        'object_detection_damoyolo_exporter':
        ['ObjectDetectionDamoyoloExporter'],
        'face_detection_scrfd_exporter': ['FaceDetectionSCRFDExporter'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
