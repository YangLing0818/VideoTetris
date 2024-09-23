# These codes are copied from modelscope revision c58451baead80d83281f063d12fb377fad415257 
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import Any, Dict

from modelscope.metainfo import Models
from modelscope.models.base import Model
from modelscope.models.builder import MODELS
from modelscope.utils.constant import Frameworks, Tasks


@MODELS.register_module(
    Tasks.speaker_verification, module_name=Models.generic_sv)
@MODELS.register_module(
    Tasks.speaker_diarization, module_name=Models.generic_sv)
class SpeakerVerification(Model):

    def __init__(self, model_dir: str, model_name: str,
                 model_config: Dict[str, Any], *args, **kwargs):
        """initialize the info of model.

        Args:
            model_dir (str): the model path.
            model_name (str): the itn model name from configuration.json
            model_config (Dict[str, Any]): the detail config about model from configuration.json
        """
        super().__init__(model_dir, model_name, model_config, *args, **kwargs)
        self.model_cfg = {
            # the recognition model dir path
            'model_workspace': model_dir,
            # the itn model name
            'model_name': model_name,
            # the am model file path
            'model_path': os.path.join(model_dir, model_name),
            # the recognition model config dict
            'model_config': model_config
        }

    def forward(self) -> Dict[str, Any]:
        """
          just return the model config

        """

        return self.model_cfg
