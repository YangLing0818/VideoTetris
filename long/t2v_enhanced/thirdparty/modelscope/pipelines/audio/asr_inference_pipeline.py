# These codes are copied from modelscope revision c58451baead80d83281f063d12fb377fad415257 
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import json
import yaml

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import WavToScp
from modelscope.utils.audio.audio_utils import (extract_pcm_from_wav,
                                                generate_scp_from_url,
                                                load_bytes_from_url,
                                                update_local_model)
from modelscope.utils.constant import Frameworks, ModelFile, Tasks
from modelscope.utils.hub import snapshot_download
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['AutomaticSpeechRecognitionPipeline']


@PIPELINES.register_module(
    Tasks.auto_speech_recognition, module_name=Pipelines.asr_inference)
class AutomaticSpeechRecognitionPipeline(Pipeline):
    """ASR Inference Pipeline
    Example:

    >>> from modelscope.pipelines import pipeline
    >>> from modelscope.utils.constant import Tasks

    >>> inference_pipeline = pipeline(
    >>>     task=Tasks.auto_speech_recognition,
    >>>     model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch')

    >>> rec_result = inference_pipeline(
    >>>     audio_in='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav')
    >>> print(rec_result)

    """

    def __init__(self,
                 model: Union[Model, str] = None,
                 preprocessor: WavToScp = None,
                 vad_model: Optional[Union[Model, str]] = None,
                 vad_model_revision: Optional[str] = None,
                 punc_model: Optional[Union[Model, str]] = None,
                 punc_model_revision: Optional[str] = None,
                 lm_model: Optional[Union[Model, str]] = None,
                 lm_model_revision: Optional[str] = None,
                 timestamp_model: Optional[Union[Model, str]] = None,
                 timestamp_model_revision: Optional[str] = None,
                 ngpu: int = 1,
                 **kwargs):
        """
        Use `model` and `preprocessor` to create an asr pipeline for prediction
        Args:
            model ('Model' or 'str'):
                The pipeline handles three types of model:

                - A model instance
                - A model local dir
                - A model id in the model hub
            preprocessor:
                (list of) Preprocessor object
            vad_model (Optional: 'Model' or 'str'):
                voice activity detection model from model hub or local
                example: 'damo/speech_fsmn_vad_zh-cn-16k-common-pytorch'
            punc_model (Optional: 'Model' or 'str'):
                punctuation model from model hub or local
                example: 'damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch'
            lm_model (Optional: 'Model' or 'str'):
                language model from model hub or local
                example: 'damo/speech_transformer_lm_zh-cn-common-vocab8404-pytorch'
            timestamp_model (Optional: 'Model' or 'str'):
                timestamp model from model hub or local
                example: 'damo/speech_timestamp_predictor-v1-16k-offline'
            output_dir('str'):
                output dir path
            batch_size('int'):
                the batch size for inference
            ngpu('int'):
                the number of gpus, 0 indicates CPU mode
            beam_size('int'):
                beam size for decoding
            ctc_weight('float'):
                the CTC weight in joint decoding
            lm_weight('float'):
                lm weight
            decoding_ind('int', defaults to 0):
                decoding ind
            decoding_mode('str', defaults to 'model1'):
                decoding mode
            vad_model_file('str'):
                vad model file
            vad_infer_config('str'):
                VAD infer configuration
            vad_cmvn_file('str'):
                global CMVN file
            punc_model_file('str'):
                punc model file
            punc_infer_config('str'):
                punc infer config
            param_dict('dict'):
                extra kwargs
        """
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        self.vad_model = vad_model
        self.vad_model_revision = vad_model_revision
        self.punc_model = punc_model
        self.punc_model_revision = punc_model_revision
        self.lm_model = lm_model
        self.lm_model_revision = lm_model_revision
        self.timestamp_model = timestamp_model
        self.timestamp_model_revision = timestamp_model_revision
        self.model_cfg = self.model.forward()

        self.cmd = self.get_cmd(kwargs, model)
        from funasr.bin import asr_inference_launch
        self.funasr_infer_modelscope = asr_inference_launch.inference_launch(
            mode=self.cmd['mode'],
            maxlenratio=self.cmd['maxlenratio'],
            minlenratio=self.cmd['minlenratio'],
            batch_size=self.cmd['batch_size'],
            beam_size=self.cmd['beam_size'],
            ngpu=ngpu,
            ctc_weight=self.cmd['ctc_weight'],
            lm_weight=self.cmd['lm_weight'],
            penalty=self.cmd['penalty'],
            log_level=self.cmd['log_level'],
            asr_train_config=self.cmd['asr_train_config'],
            asr_model_file=self.cmd['asr_model_file'],
            cmvn_file=self.cmd['cmvn_file'],
            lm_file=self.cmd['lm_file'],
            token_type=self.cmd['token_type'],
            key_file=self.cmd['key_file'],
            lm_train_config=self.cmd['lm_train_config'],
            bpemodel=self.cmd['bpemodel'],
            allow_variable_data_keys=self.cmd['allow_variable_data_keys'],
            output_dir=self.cmd['output_dir'],
            dtype=self.cmd['dtype'],
            seed=self.cmd['seed'],
            ngram_weight=self.cmd['ngram_weight'],
            nbest=self.cmd['nbest'],
            num_workers=self.cmd['num_workers'],
            vad_infer_config=self.cmd['vad_infer_config'],
            vad_model_file=self.cmd['vad_model_file'],
            vad_cmvn_file=self.cmd['vad_cmvn_file'],
            punc_model_file=self.cmd['punc_model_file'],
            punc_infer_config=self.cmd['punc_infer_config'],
            timestamp_model_file=self.cmd['timestamp_model_file'],
            timestamp_infer_config=self.cmd['timestamp_infer_config'],
            timestamp_cmvn_file=self.cmd['timestamp_cmvn_file'],
            outputs_dict=self.cmd['outputs_dict'],
            param_dict=self.cmd['param_dict'],
            token_num_relax=self.cmd['token_num_relax'],
            decoding_ind=self.cmd['decoding_ind'],
            decoding_mode=self.cmd['decoding_mode'],
            **kwargs,
        )

    def __call__(self,
                 audio_in: Union[str, bytes],
                 audio_fs: int = None,
                 recog_type: str = None,
                 audio_format: str = None,
                 output_dir: str = None,
                 param_dict: dict = None,
                 **kwargs) -> Dict[str, Any]:
        from funasr.utils import asr_utils
        """
        Decoding the input audios
        Args:
            audio_in('str' or 'bytes'):
                - A string containing a local path to a wav file
                - A string containing a local path to a scp
                - A string containing a wav url
                - A bytes input
            audio_fs('int'):
                frequency of sample
            recog_type('str'):
                recog type
            audio_format('str'):
                audio format
            output_dir('str'):
                output dir
            param_dict('dict'):
                extra kwargs
        Return:
            A dictionary of result or a list of dictionary of result.

            The dictionary contain the following keys:
            - **text** ('str') --The asr result.
        """

        # code base
        # code_base = self.cmd['code_base']
        self.recog_type = recog_type
        self.audio_format = audio_format
        self.audio_fs = None
        checking_audio_fs = None
        self.raw_inputs = None
        if output_dir is not None:
            self.cmd['output_dir'] = output_dir
        self.cmd['param_dict'] = param_dict

        if isinstance(audio_in, str):
            # for funasr code, generate wav.scp from url or local path
            if audio_in.startswith('http') or os.path.isfile(audio_in):
                self.audio_in, self.raw_inputs = generate_scp_from_url(
                    audio_in)
            else:
                raise FileNotFoundError(
                    f'file {audio_in} NOT FOUND, please CHECK!')
        elif isinstance(audio_in, bytes):
            self.audio_in = audio_in
            self.raw_inputs = None
        else:
            import numpy
            import torch
            if isinstance(audio_in, torch.Tensor):
                self.audio_in = None
                self.raw_inputs = audio_in
            elif isinstance(audio_in, numpy.ndarray):
                self.audio_in = None
                self.raw_inputs = audio_in

        # set the sample_rate of audio_in if checking_audio_fs is valid
        if checking_audio_fs is not None:
            self.audio_fs = checking_audio_fs

        if recog_type is None or audio_format is None:
            self.recog_type, self.audio_format, self.audio_in = asr_utils.type_checking(
                audio_in=self.audio_in,
                recog_type=recog_type,
                audio_format=audio_format)

        if hasattr(asr_utils,
                   'sample_rate_checking') and self.audio_in is not None:
            checking_audio_fs = asr_utils.sample_rate_checking(
                self.audio_in, self.audio_format)
            if checking_audio_fs is not None:
                self.audio_fs = checking_audio_fs
        if audio_fs is not None:
            self.cmd['fs']['audio_fs'] = audio_fs
        else:
            self.cmd['fs']['audio_fs'] = self.audio_fs

        output = self.preprocessor.forward(self.model_cfg, self.recog_type,
                                           self.audio_format, self.audio_in,
                                           self.audio_fs, self.cmd)
        output = self.forward(output, **kwargs)
        rst = self.postprocess(output)
        return rst

    def get_cmd(self, extra_args, model_path) -> Dict[str, Any]:
        if self.preprocessor is None:
            self.preprocessor = WavToScp()

        outputs = self.preprocessor.config_checking(self.model_cfg)
        # generate asr inference command
        cmd = {
            'maxlenratio': 0.0,
            'minlenratio': 0.0,
            'batch_size': 1,
            'beam_size': 1,
            'ngpu': 1,
            'ctc_weight': 0.0,
            'lm_weight': 0.0,
            'penalty': 0.0,
            'log_level': 'ERROR',
            'asr_train_config': None,
            'asr_model_file': outputs['am_model_path'],
            'cmvn_file': None,
            'lm_train_config': None,
            'lm_file': None,
            'token_type': None,
            'key_file': None,
            'word_lm_train_config': None,
            'bpemodel': None,
            'allow_variable_data_keys': False,
            'output_dir': None,
            'dtype': 'float32',
            'seed': 0,
            'ngram_weight': 0.9,
            'nbest': 1,
            'num_workers': 0,
            'vad_infer_config': None,
            'vad_model_file': None,
            'vad_cmvn_file': None,
            'time_stamp_writer': True,
            'punc_infer_config': None,
            'punc_model_file': None,
            'timestamp_infer_config': None,
            'timestamp_model_file': None,
            'timestamp_cmvn_file': None,
            'outputs_dict': True,
            'param_dict': None,
            'model_type': outputs['model_type'],
            'idx_text': '',
            'sampled_ids': 'seq2seq/sampled_ids',
            'sampled_lengths': 'seq2seq/sampled_lengths',
            'lang': 'zh-cn',
            'code_base': outputs['code_base'],
            'mode': outputs['mode'],
            'fs': {
                'model_fs': None,
                'audio_fs': None
            }
        }

        frontend_conf = None
        token_num_relax = None
        decoding_ind = None
        decoding_mode = None
        if os.path.exists(outputs['am_model_config']):
            config_file = open(outputs['am_model_config'], encoding='utf-8')
            root = yaml.full_load(config_file)
            config_file.close()
            if 'frontend_conf' in root:
                frontend_conf = root['frontend_conf']
        if os.path.exists(outputs['asr_model_config']):
            config_file = open(outputs['asr_model_config'], encoding='utf-8')
            root = yaml.full_load(config_file)
            config_file.close()
            if 'token_num_relax' in root:
                token_num_relax = root['token_num_relax']
            if 'decoding_ind' in root:
                decoding_ind = root['decoding_ind']
            if 'decoding_mode' in root:
                decoding_mode = root['decoding_mode']

            cmd['beam_size'] = root['beam_size']
            cmd['penalty'] = root['penalty']
            cmd['maxlenratio'] = root['maxlenratio']
            cmd['minlenratio'] = root['minlenratio']
            cmd['ctc_weight'] = root['ctc_weight']
            cmd['lm_weight'] = root['lm_weight']
        cmd['asr_train_config'] = outputs['am_model_config']
        cmd['lm_file'] = outputs['lm_model_path']
        cmd['lm_train_config'] = outputs['lm_model_config']
        cmd['batch_size'] = outputs['model_config']['batch_size']
        cmd['frontend_conf'] = frontend_conf
        if frontend_conf is not None and 'fs' in frontend_conf:
            cmd['fs']['model_fs'] = frontend_conf['fs']
        cmd['token_num_relax'] = token_num_relax
        cmd['decoding_ind'] = decoding_ind
        cmd['decoding_mode'] = decoding_mode
        if outputs.__contains__('mvn_file'):
            cmd['cmvn_file'] = outputs['mvn_file']
        model_config = self.model_cfg['model_config']
        if model_config.__contains__('vad_model') and self.vad_model != '':
            self.vad_model = model_config['vad_model']
        if model_config.__contains__('vad_model_revision'):
            self.vad_model_revision = model_config['vad_model_revision']
        if model_config.__contains__('punc_model') and self.punc_model != '':
            self.punc_model = model_config['punc_model']
        if model_config.__contains__('punc_model_revision'):
            self.punc_model_revision = model_config['punc_model_revision']
        if model_config.__contains__(
                'timestamp_model') and self.timestamp_model != '':
            self.timestamp_model = model_config['timestamp_model']
        if model_config.__contains__('timestamp_model_revision'):
            self.timestamp_model_revision = model_config[
                'timestamp_model_revision']
        update_local_model(model_config, model_path, extra_args)
        self.load_vad_model(cmd)
        self.load_punc_model(cmd)
        self.load_lm_model(cmd)
        self.load_timestamp_model(cmd)

        user_args_dict = [
            'output_dir',
            'batch_size',
            'mode',
            'ngpu',
            'beam_size',
            'ctc_weight',
            'lm_weight',
            'decoding_ind',
            'decoding_mode',
            'vad_model_file',
            'vad_infer_config',
            'vad_cmvn_file',
            'punc_model_file',
            'punc_infer_config',
            'param_dict',
        ]

        for user_args in user_args_dict:
            if user_args in extra_args:
                if extra_args.get(user_args) is not None:
                    cmd[user_args] = extra_args[user_args]
                del extra_args[user_args]

        return cmd

    def load_vad_model(self, cmd):
        if self.vad_model is not None and self.vad_model != '':
            if os.path.exists(self.vad_model):
                vad_model = self.vad_model
            else:
                vad_model = snapshot_download(
                    self.vad_model, revision=self.vad_model_revision)
            logger.info('loading vad model from {0} ...'.format(vad_model))
            config_path = os.path.join(vad_model, ModelFile.CONFIGURATION)
            model_cfg = json.loads(open(config_path).read())
            model_dir = os.path.dirname(config_path)
            cmd['vad_model_file'] = os.path.join(
                model_dir,
                model_cfg['model']['model_config']['vad_model_name'])
            cmd['vad_infer_config'] = os.path.join(
                model_dir,
                model_cfg['model']['model_config']['vad_model_config'])
            cmd['vad_cmvn_file'] = os.path.join(
                model_dir, model_cfg['model']['model_config']['vad_mvn_file'])
            if 'vad' not in cmd['mode']:
                cmd['mode'] = cmd['mode'] + '_vad'

    def load_punc_model(self, cmd):
        if self.punc_model is not None and self.punc_model != '':
            if os.path.exists(self.punc_model):
                punc_model = self.punc_model
            else:
                punc_model = snapshot_download(
                    self.punc_model, revision=self.punc_model_revision)
            logger.info(
                'loading punctuation model from {0} ...'.format(punc_model))
            config_path = os.path.join(punc_model, ModelFile.CONFIGURATION)
            model_cfg = json.loads(open(config_path).read())
            model_dir = os.path.dirname(config_path)
            cmd['punc_model_file'] = os.path.join(
                model_dir, model_cfg['model']['punc_model_name'])
            cmd['punc_infer_config'] = os.path.join(
                model_dir,
                model_cfg['model']['punc_model_config']['punc_config'])
            if 'punc' not in cmd['mode']:
                cmd['mode'] = cmd['mode'] + '_punc'

    def load_lm_model(self, cmd):
        if self.lm_model is not None and self.lm_model != '':
            if os.path.exists(self.lm_model):
                lm_model = self.lm_model
            else:
                lm_model = snapshot_download(
                    self.lm_model, revision=self.lm_model_revision)
            logger.info('loading language model from {0} ...'.format(lm_model))
            config_path = os.path.join(lm_model, ModelFile.CONFIGURATION)
            model_cfg = json.loads(open(config_path).read())
            model_dir = os.path.dirname(config_path)
            cmd['lm_file'] = os.path.join(
                model_dir, model_cfg['model']['model_config']['lm_model_name'])
            cmd['lm_train_config'] = os.path.join(
                model_dir,
                model_cfg['model']['model_config']['lm_model_config'])

    # FIXME
    def load_timestamp_model(self, cmd):
        if self.timestamp_model is not None and self.timestamp_model != '':
            if os.path.exists(self.timestamp_model):
                timestamp_model = self.timestamp_model
            else:
                timestamp_model = snapshot_download(
                    self.timestamp_model,
                    revision=self.timestamp_model_revision)
            logger.info(
                'loading timestamp model from {0} ...'.format(timestamp_model))
            config_path = os.path.join(timestamp_model,
                                       ModelFile.CONFIGURATION)
            model_cfg = json.loads(open(config_path).read())
            model_dir = os.path.dirname(config_path)
            cmd['timestamp_model_file'] = os.path.join(
                model_dir,
                model_cfg['model']['model_config']['timestamp_model_file'])
            cmd['timestamp_infer_config'] = os.path.join(
                model_dir,
                model_cfg['model']['model_config']['timestamp_infer_config'])
            cmd['timestamp_cmvn_file'] = os.path.join(
                model_dir,
                model_cfg['model']['model_config']['timestamp_cmvn_file'])

    def forward(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Decoding
        """

        logger.info(f"Decoding with {inputs['audio_format']} files ...")

        data_cmd: Sequence[Tuple[str, str, str]]
        if isinstance(self.audio_in, bytes):
            data_cmd = [self.audio_in, 'speech', 'bytes']
        elif isinstance(self.audio_in, str):
            data_cmd = [self.audio_in, 'speech', 'sound']
        elif self.raw_inputs is not None:
            data_cmd = None

        # generate asr inference command
        self.cmd['name_and_type'] = data_cmd
        self.cmd['raw_inputs'] = self.raw_inputs
        self.cmd['audio_in'] = self.audio_in

        inputs['asr_result'] = self.run_inference(self.cmd, **kwargs)

        return inputs

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """process the asr results
        """
        from funasr.utils import asr_utils

        logger.info('Computing the result of ASR ...')

        rst = {}

        # single wav or pcm task
        if inputs['recog_type'] == 'wav':
            if 'asr_result' in inputs and len(inputs['asr_result']) > 0:
                for key, value in inputs['asr_result'][0].items():
                    if key == 'value':
                        if len(value) > 0:
                            rst[OutputKeys.TEXT] = value
                    elif key != 'key':
                        rst[key] = value

        # run with datasets, and audio format is waveform or kaldi_ark or tfrecord
        elif inputs['recog_type'] != 'wav':
            inputs['reference_list'] = self.ref_list_tidy(inputs)

            inputs['datasets_result'] = asr_utils.compute_wer(
                hyp_list=inputs['asr_result'],
                ref_list=inputs['reference_list'])

        else:
            raise ValueError('recog_type and audio_format are mismatching')

        if 'datasets_result' in inputs:
            rst[OutputKeys.TEXT] = inputs['datasets_result']

        return rst

    def ref_list_tidy(self, inputs: Dict[str, Any]) -> List[Any]:
        ref_list = []

        if inputs['audio_format'] == 'tfrecord':
            # should assemble idx + txt
            with open(inputs['reference_text'], 'r', encoding='utf-8') as r:
                text_lines = r.readlines()

            with open(inputs['idx_text'], 'r', encoding='utf-8') as i:
                idx_lines = i.readlines()

            j: int = 0
            while j < min(len(text_lines), len(idx_lines)):
                idx_str = idx_lines[j].strip()
                text_str = text_lines[j].strip().replace(' ', '')
                item = {'key': idx_str, 'value': text_str}
                ref_list.append(item)
                j += 1

        else:
            # text contain idx + sentence
            with open(inputs['reference_text'], 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines:
                line_item = line.split(None, 1)
                if len(line_item) > 1:
                    item = {
                        'key': line_item[0],
                        'value': line_item[1].strip('\n')
                    }
                    ref_list.append(item)

        return ref_list

    def run_inference(self, cmd, **kwargs):
        asr_result = self.funasr_infer_modelscope(cmd['name_and_type'],
                                                  cmd['raw_inputs'],
                                                  cmd['output_dir'], cmd['fs'],
                                                  cmd['param_dict'], **kwargs)

        return asr_result
