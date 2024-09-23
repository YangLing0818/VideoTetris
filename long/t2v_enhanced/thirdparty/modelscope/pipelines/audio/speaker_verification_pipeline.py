# These codes are copied from modelscope revision c58451baead80d83281f063d12fb377fad415257 
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
from typing import Any, Dict, List, Sequence, Tuple, Union

import yaml

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.audio.audio_utils import (generate_scp_for_sv,
                                                generate_sv_scp_from_url,
                                                update_local_model)
from modelscope.utils.constant import Frameworks, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['SpeakerVerificationPipeline']


@PIPELINES.register_module(
    Tasks.speaker_verification, module_name=Pipelines.sv_inference)
class SpeakerVerificationPipeline(Pipeline):
    """Speaker Verification Inference Pipeline
    use `model` to create a Speaker Verification pipeline.

    Args:
        model (SpeakerVerificationPipeline): A model instance, or a model local dir, or a model id in the model hub.
        kwargs (dict, `optional`):
            Extra kwargs passed into the preprocessor's constructor.
    Examples:
        >>> from modelscope.pipelines import pipeline
        >>> pipeline_sv = pipeline(
        >>>    task=Tasks.speaker_verification, model='damo/speech_xvector_sv-zh-cn-cnceleb-16k-spk3465-pytorch')
        >>> audio_in=('sv_example_enroll.wav', 'sv_example_same.wav')
        >>> print(pipeline_sv(audio_in))
        >>> # {'label': ['Same', 'Different'], 'scores': [0.8540488358969999, 0.14595116410300013]}

    """

    def __init__(self,
                 model: Union[Model, str] = None,
                 ngpu: int = 1,
                 **kwargs):
        """use `model` to create an asr pipeline for prediction
        """
        super().__init__(model=model, **kwargs)
        self.model_cfg = self.model.forward()
        self.cmd = self.get_cmd(kwargs, model)

        from funasr.bin import sv_inference_launch
        self.funasr_infer_modelscope = sv_inference_launch.inference_launch(
            mode=self.cmd['mode'],
            output_dir=self.cmd['output_dir'],
            batch_size=self.cmd['batch_size'],
            dtype=self.cmd['dtype'],
            ngpu=ngpu,
            seed=self.cmd['seed'],
            num_workers=self.cmd['num_workers'],
            log_level=self.cmd['log_level'],
            key_file=self.cmd['key_file'],
            sv_train_config=self.cmd['sv_train_config'],
            sv_model_file=self.cmd['sv_model_file'],
            model_tag=self.cmd['model_tag'],
            allow_variable_data_keys=self.cmd['allow_variable_data_keys'],
            streaming=self.cmd['streaming'],
            embedding_node=self.cmd['embedding_node'],
            sv_threshold=self.cmd['sv_threshold'],
            param_dict=self.cmd['param_dict'],
            **kwargs,
        )

    def __call__(self,
                 audio_in: Union[tuple, str, Any] = None,
                 output_dir: str = None,
                 param_dict: dict = None) -> Dict[str, Any]:
        if len(audio_in) == 0:
            raise ValueError('The input of sv should not be null.')
        else:
            self.audio_in = audio_in
        if output_dir is not None:
            self.cmd['output_dir'] = output_dir
        self.cmd['param_dict'] = param_dict

        output = self.forward(self.audio_in)
        result = self.postprocess(output)
        return result

    def postprocess(self, inputs: list) -> Dict[str, Any]:
        """Postprocessing
        """
        rst = {}
        for i in range(len(inputs)):
            # for single input, re-formate the output
            # audio_in:
            #   list/tuple: return speaker verification scores
            #   single wav/bytes: return speaker embedding
            if len(inputs) == 1 and i == 0:
                if isinstance(self.audio_in, tuple) or isinstance(
                        self.audio_in, list):
                    score = inputs[0]['value']
                    rst[OutputKeys.LABEL] = ['Same', 'Different']
                    rst[OutputKeys.SCORES] = [score / 100.0, 1 - score / 100.0]
                else:
                    embedding = inputs[0]['value']
                    rst[OutputKeys.SPK_EMBEDDING] = embedding
            else:
                # for multiple inputs
                rst[inputs[i]['key']] = inputs[i]['value']
        return rst

    def get_cmd(self, extra_args, model_path) -> Dict[str, Any]:
        # generate asr inference command
        mode = self.model_cfg['model_config']['mode']
        sv_model_path = self.model_cfg['model_path']
        sv_model_config = os.path.join(
            self.model_cfg['model_workspace'],
            self.model_cfg['model_config']['sv_model_config'])
        update_local_model(self.model_cfg['model_config'], model_path,
                           extra_args)
        cmd = {
            'mode': mode,
            'output_dir': None,
            'batch_size': 1,
            'dtype': 'float32',
            'ngpu': 1,  # 0: only CPU, ngpu>=1: gpu number if cuda is available
            'seed': 0,
            'num_workers': 0,
            'log_level': 'ERROR',
            'key_file': None,
            'sv_model_file': sv_model_path,
            'sv_train_config': sv_model_config,
            'model_tag': None,
            'allow_variable_data_keys': True,
            'streaming': False,
            'embedding_node': 'resnet1_dense',
            'sv_threshold': 0.9465,
            'param_dict': None,
        }
        user_args_dict = [
            'output_dir',
            'batch_size',
            'ngpu',
            'embedding_node',
            'sv_threshold',
            'log_level',
            'allow_variable_data_keys',
            'streaming',
            'num_workers',
            'param_dict',
        ]

        # re-write the config with configure.json
        for user_args in user_args_dict:
            if (user_args in self.model_cfg['model_config']
                    and self.model_cfg['model_config'][user_args] is not None):
                if isinstance(cmd[user_args], dict) and isinstance(
                        self.model_cfg['model_config'][user_args], dict):
                    cmd[user_args].update(
                        self.model_cfg['model_config'][user_args])
                else:
                    cmd[user_args] = self.model_cfg['model_config'][user_args]

        # rewrite the config with user args
        for user_args in user_args_dict:
            if user_args in extra_args:
                if extra_args.get(user_args) is not None:
                    if isinstance(cmd[user_args], dict) and isinstance(
                            extra_args[user_args], dict):
                        cmd[user_args].update(extra_args[user_args])
                    else:
                        cmd[user_args] = extra_args[user_args]
                del extra_args[user_args]

        return cmd

    def forward(self, audio_in: Union[tuple, str, Any] = None) -> list:
        """Decoding
        """
        # log  file_path/url or tuple (str, str)
        if isinstance(audio_in, str) or \
                (isinstance(audio_in, tuple) and all(isinstance(item, str) for item in audio_in)):
            logger.info(f'Speaker Verification Processing: {audio_in} ...')
        else:
            logger.info(
                f'Speaker Verification Processing: {str(audio_in)[:100]} ...')

        data_cmd, raw_inputs = None, None
        if isinstance(audio_in, tuple) or isinstance(audio_in, list):
            # generate audio_scp
            assert len(audio_in) == 2
            if isinstance(audio_in[0], str):
                # for scp inputs
                if len(audio_in[0].split(',')) == 3 and audio_in[0].split(
                        ',')[0].endswith('.scp'):
                    if len(audio_in[1].split(',')) == 3 and audio_in[1].split(
                            ',')[0].endswith('.scp'):
                        data_cmd = [
                            tuple(audio_in[0].split(',')),
                            tuple(audio_in[1].split(','))
                        ]
                # for single-file inputs
                else:
                    audio_scp_1, audio_scp_2 = generate_sv_scp_from_url(
                        audio_in)
                    if isinstance(audio_scp_1, bytes) and isinstance(
                            audio_scp_2, bytes):
                        data_cmd = [(audio_scp_1, 'speech', 'bytes'),
                                    (audio_scp_2, 'ref_speech', 'bytes')]
                    else:
                        data_cmd = [(audio_scp_1, 'speech', 'sound'),
                                    (audio_scp_2, 'ref_speech', 'sound')]
            # for raw bytes inputs
            elif isinstance(audio_in[0], bytes):
                data_cmd = [(audio_in[0], 'speech', 'bytes'),
                            (audio_in[1], 'ref_speech', 'bytes')]
            else:
                raise TypeError('Unsupported data type.')
        else:
            if isinstance(audio_in, str):
                # for scp inputs
                if len(audio_in.split(',')) == 3:
                    data_cmd = [audio_in.split(',')]
                # for single-file inputs
                else:
                    audio_scp = generate_scp_for_sv(audio_in)
                    if isinstance(audio_scp, bytes):
                        data_cmd = [(audio_scp, 'speech', 'bytes')]
                    else:
                        data_cmd = [(audio_scp, 'speech', 'sound')]
            # for raw bytes
            elif isinstance(audio_in, bytes):
                data_cmd = [(audio_in, 'speech', 'bytes')]
            # for ndarray and tensor inputs
            else:
                import torch
                import numpy as np
                if isinstance(audio_in, torch.Tensor):
                    raw_inputs = audio_in
                elif isinstance(audio_in, np.ndarray):
                    raw_inputs = audio_in
                else:
                    raise TypeError('Unsupported data type.')

        self.cmd['name_and_type'] = data_cmd
        self.cmd['raw_inputs'] = raw_inputs
        result = self.run_inference(self.cmd)

        return result

    def run_inference(self, cmd):
        if self.framework == Frameworks.torch:
            sv_result = self.funasr_infer_modelscope(
                data_path_and_name_and_type=cmd['name_and_type'],
                raw_inputs=cmd['raw_inputs'],
                output_dir_v2=cmd['output_dir'],
                param_dict=cmd['param_dict'])
        else:
            raise ValueError('model type is mismatching')

        return sv_result
