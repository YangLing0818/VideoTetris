# These codes are copied from modelscope revision c58451baead80d83281f063d12fb377fad415257 
# Copyright (c) Alibaba, Inc. and its affiliates.
import ast
import base64
import importlib
import inspect
from io import BytesIO
from typing import Any
from urllib.parse import urlparse

import numpy as np

from modelscope.hub.api import HubApi
from modelscope.hub.errors import NotExistError
from modelscope.hub.file_download import model_file_download
from modelscope.outputs.outputs import (TASK_OUTPUTS, OutputKeys, OutputTypes,
                                        OutputTypeSchema)
from modelscope.pipeline_inputs import (INPUT_TYPE, INPUT_TYPE_SCHEMA,
                                        TASK_INPUTS, InputType)
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Pipeline
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()
"""Support webservice integration pipeline。

This module provides a support library when webservice uses pipeline,
converts webservice input into pipeline input, and converts pipeline
output into webservice output, which automatically encodes and
decodes relevant fields.

Example:
    # create pipeine instance and pipeline information, save it to app
    pipeline_instance = create_pipeline('damo/cv_gpen_image-portrait-enhancement', 'v1.0.0')
    pipeline_info = get_pipeline_information_by_pipeline(pipeline_instance)
    app.state.pipeline = pipeline_instance
    app.state.pipeline_info = pipeline_info

    # for service schema request.
    pipeline_info = request.app.state.pipeline_info
    return pipeline_info.schema

    # for service call request.
    def inference(request: Request):
        pipeline_service = request.app.state.pipeline
        pipeline_info = request.app.state.pipeline_info
        request_json = await request.json()
        result = call_pipeline_with_json(pipeline_info,
                            pipeline_service,
                            request_json)
        # convert output to json, if binary field, we need encoded.
        output = pipeline_output_to_service_base64_output(pipeline_info.task_name, result)
        return output
Todo:
    * Support more service input type, such as form.

"""


def create_pipeline(model_id: str, revision: str):
    model_configuration_file = model_file_download(
        model_id=model_id,
        file_path=ModelFile.CONFIGURATION,
        revision=revision)
    cfg = Config.from_file(model_configuration_file)
    return pipeline(task=cfg.task, model=model_id, model_revision=revision)


def get_class_user_attributes(cls):
    attributes = inspect.getmembers(cls, lambda a: not (inspect.isroutine(a)))
    user_attributes = [
        a for a in attributes
        if (not (a[0].startswith('__') and a[0].endswith('__')))
    ]
    return user_attributes


def get_input_type(task_inputs: Any):
    """Get task input schema.

    Args:
        task_name (str): The task name.
    """
    if isinstance(task_inputs, str):  # no input key
        input_type = INPUT_TYPE[task_inputs]
        return input_type
    elif isinstance(task_inputs, tuple) or isinstance(task_inputs, list):
        for item in task_inputs:
            if isinstance(item,
                          dict):  # for list, server only support dict format.
                return get_input_type(item)
            else:
                continue
    elif isinstance(task_inputs, dict):
        input_info = {}  # key input key, value input type
        for k, v in task_inputs.items():
            input_info[k] = get_input_type(v)
        return input_info
    else:
        raise ValueError(f'invalid input_type definition {task_inputs}')


def get_input_schema(task_name: str, input_type: type):
    """Get task input schema.

    Args:
        task_name (str): The task name.
        input_type (type): The input type
    """
    if input_type is None:
        task_inputs = TASK_INPUTS[task_name]
        if isinstance(task_inputs,
                      str):  # only one input field, key is task_inputs
            return {
                'type': 'object',
                'properties': {
                    task_inputs: INPUT_TYPE_SCHEMA[task_inputs]
                }
            }
    else:
        task_inputs = input_type

    if isinstance(task_inputs, str):  # no input key
        return INPUT_TYPE_SCHEMA[task_inputs]
    elif input_type is None and isinstance(task_inputs, list):
        for item in task_inputs:
            # for list, server only support dict format.
            if isinstance(item, dict):
                return get_input_schema(None, item)
    elif isinstance(task_inputs, tuple) or isinstance(task_inputs, list):
        input_schema = {'type': 'array', 'items': {}}
        for item in task_inputs:
            if isinstance(item, dict):
                item_schema = get_input_schema(None, item)
                input_schema['items']['type'] = item_schema
                return input_schema
            else:
                input_schema['items'] = INPUT_TYPE_SCHEMA[item]
                return input_schema

    elif isinstance(task_inputs, dict):
        input_schema = {
            'type': 'object',
            'properties': {}
        }  # key input key, value input type
        for k, v in task_inputs.items():
            input_schema['properties'][k] = get_input_schema(None, v)
        return input_schema
    else:
        raise ValueError(f'invalid input_type definition {task_inputs}')


def get_output_schema(task_name: str):
    """Get task output schema.

    Args:
        task_name (str): The task name.
    """
    task_outputs = TASK_OUTPUTS[task_name]
    output_schema = {'type': 'object', 'properties': {}}
    if not isinstance(task_outputs, list):
        raise ValueError('TASK_OUTPUTS for %s is not list.' % task_name)
    else:
        for output_key in task_outputs:
            output_schema['properties'][output_key] = OutputTypeSchema[
                output_key]
    return output_schema


def get_input_info(task_name: str):
    task_inputs = TASK_INPUTS[task_name]
    if isinstance(task_inputs, str):  # no input key default input key input
        input_type = INPUT_TYPE[task_inputs]
        return input_type
    elif isinstance(task_inputs, tuple):
        return task_inputs
    elif isinstance(task_inputs, list):
        for item in task_inputs:
            if isinstance(item,
                          dict):  # for list, server only support dict format.
                return {'input': get_input_type(item)}
            else:
                continue
    elif isinstance(task_inputs, dict):
        input_info = {}  # key input key, value input type
        for k, v in task_inputs.items():
            input_info[k] = get_input_type(v)
        return {'input': input_info}
    else:
        raise ValueError(f'invalid input_type definition {task_inputs}')


def get_output_info(task_name: str):
    output_keys = TASK_OUTPUTS[task_name]
    output_type = {}
    if not isinstance(output_keys, list):
        raise ValueError('TASK_OUTPUTS for %s is not list.' % task_name)
    else:
        for output_key in output_keys:
            output_type[output_key] = OutputTypes[output_key]
    return output_type


def get_task_io_info(task_name: str):
    """Get task input output schema.

    Args:
        task_name (str): The task name.
    """
    tasks = get_class_user_attributes(Tasks)
    task_exist = False
    for key, value in tasks:
        if key == task_name or value == task_name:
            task_exist = True
            break
    if not task_exist:
        return None, None

    task_inputs = get_input_info(task_name)
    task_outputs = get_output_info(task_name)

    return task_inputs, task_outputs


def process_arg_type_annotation(arg, default_value):
    if arg.annotation is not None:
        if isinstance(arg.annotation, ast.Subscript):
            return arg.arg, arg.annotation.value.id
        elif isinstance(arg.annotation, ast.Name):
            return arg.arg, arg.annotation.id
        elif isinstance(arg.annotation, ast.Attribute):
            return arg.arg, arg.annotation.attr
        else:
            raise Exception('Invalid annotation: %s' % arg.annotation)
    else:
        if default_value is not None:
            return arg.arg, type(default_value).__name__
        # Irregular, assuming no type hint no default value type is object
        logger.warning('arg: %s has no data type annotation, use default!' %
                       (arg.arg))
        return arg.arg, 'object'


def process_args(args):
    arguments = []
    # name, type, has_default, default
    n_args = len(args.args)
    n_args_default = len(args.defaults)
    # no default
    for arg in args.args[0:n_args - n_args_default]:
        if arg.arg == 'self':
            continue
        else:
            arg_name, arg_type = process_arg_type_annotation(arg, None)
            arguments.append((arg_name, arg_type, False, None))

    # process defaults arg.
    for arg, dft in zip(args.args[n_args - n_args_default:], args.defaults):
        # compatible with python3.7 ast.Num no value.
        value = dft.value if hasattr(dft, 'value') else dft.n
        arg_name, arg_type = process_arg_type_annotation(arg, value)
        arguments.append((arg_name, arg_type, True, value))

    # kwargs
    n_kwargs = len(args.kwonlyargs)
    n_kwargs_default = len(args.kw_defaults)
    for kwarg in args.kwonlyargs[0:n_kwargs - n_kwargs_default]:
        arg_name, arg_type = process_arg_type_annotation(kwarg)
        arguments.append((arg_name, arg_type, False, None))

    for kwarg, dft in zip(args.kwonlyargs[n_kwargs - n_kwargs_default:],
                          args.kw_defaults):
        arg_name, arg_type = process_arg_type_annotation(kwarg)
        arguments.append((arg_name, arg_type, True, dft.value))
    return arguments


class PipelineClassAnalyzer(ast.NodeVisitor):
    """Analysis pipeline class define get inputs and parameters.
    """

    def __init__(self) -> None:
        super().__init__()
        self.parameters = []
        self.has_call = False
        self.preprocess_parameters = []
        self.has_preprocess = False
        self.has_postprocess = False
        self.has_forward = False
        self.forward_parameters = []
        self.postprocess_parameters = []
        self.lineno = 0
        self.end_lineno = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        if node.name == '__call__':
            self.parameters = process_args(node.args)
            self.has_call = True
        if node.name == 'preprocess':
            self.preprocess_parameters = process_args(node.args)
            self.has_preprocess = True
        elif node.name == 'postprocess':
            self.postprocess_parameters = process_args(node.args)
            self.has_postprocess = True
        elif node.name == 'forward':
            self.forward_parameters = process_args(node.args)
            self.has_forward = True

    def get_input_parameters(self):
        if self.has_call:
            # custom define __call__ inputs and parameter are control by the
            # custom __call__, all parameter is input.
            return self.parameters, None
        parameters = []
        if self.has_preprocess:
            parameters.extend(self.preprocess_parameters[1:])
        if self.has_forward:
            parameters.extend(self.forward_parameters[1:])
        if self.has_postprocess:
            parameters.extend(self.postprocess_parameters[1:])

        if len(parameters) > 0:
            return None, parameters
        else:
            return None, []


class AnalysisSourceFileRegisterModules(ast.NodeVisitor):
    """Get register_module call of the python source file.


    Args:
        ast (NodeVisitor): The ast node.

    Examples:
        >>> with open(source_file_path, "rb") as f:
        >>>     src = f.read()
        >>>     analyzer = AnalysisSourceFileRegisterModules(source_file_path)
        >>>     analyzer.visit(ast.parse(src, filename=source_file_path))
    """

    def __init__(self, source_file_path, class_name) -> None:
        super().__init__()
        self.source_file_path = source_file_path
        self.class_name = class_name
        self.class_define = None

    def visit_ClassDef(self, node: ast.ClassDef):
        if node.name == self.class_name:
            self.class_define = node


def get_pipeline_input_parameters(
    source_file_path: str,
    class_name: str,
):
    """Get pipeline input and parameter

    Args:
        source_file_path (str): The pipeline source code path
        class_name (str): The pipeline class name
    """
    with open(source_file_path, 'rb') as f:
        src = f.read()
        analyzer = AnalysisSourceFileRegisterModules(source_file_path,
                                                     class_name)
        analyzer.visit(
            ast.parse(
                src,
                filename=source_file_path,
                # python3.7 no type_comments parameter .
                # type_comments=True
            ))
        clz = PipelineClassAnalyzer()
        clz.visit(analyzer.class_define)
        input, pipeline_parameters = clz.get_input_parameters()
        # remove the first input parameter, the input is defined by task.
        return input, pipeline_parameters


meta_type_schema_map = {
    # For parameters, current only support types.
    'str': 'string',
    'int': 'integer',
    'float': 'number',
    'bool': 'boolean',
    'Dict': 'object',
    'dict': 'object',
    'list': 'array',
    'List': 'array',
    'Union': 'object',
    'Input': 'object',
    'object': 'object',
}


def generate_pipeline_parameters_schema(parameters):
    parameters_schema = {'type': 'object', 'properties': {}}
    if len(parameters) == 0:
        return {}
    for param in parameters:
        name, param_type, has_default, default_value = param
        # 'max_length': ('int', True, 1024)
        prop = {'type': meta_type_schema_map[param_type]}
        if has_default:
            prop['default'] = default_value
        parameters_schema['properties'][name] = prop
    return parameters_schema


def get_pipeline_information_by_pipeline(pipeline: Pipeline, ):
    """Get pipeline input output schema.

    Args:
        pipeline (Pipeline): The pipeline object.
    """
    task_name = pipeline.group_key
    pipeline_class = pipeline.__class__.__name__
    spec = importlib.util.find_spec(pipeline.__module__)
    pipeline_file_path = spec.origin
    info = PipelineInfomation(task_name, pipeline_class, pipeline_file_path)
    return info


class PipelineInfomation():
    """Analyze pipeline information, task_name, schema.
    """

    def __init__(self, task_name: str, class_name, source_path):
        self._task_name = task_name
        self._class_name = class_name
        self._source_path = source_path
        self._is_custom_call_method = False
        self._analyze()

    def _analyze(self):
        input, parameters = get_pipeline_input_parameters(
            self._source_path, self._class_name)
        if input is not None:  # custom pipeline __call__ asr_inferrnce_pipeline
            self._is_custom_call_method = True
            self._input_schema = generate_pipeline_parameters_schema(input)
            self._input_schema[
                'description'] = 'For binary input such as image audio video, only url is supported.'
            self._parameters_schema = {}
            self._output_schema = {
                'type': 'object',
            }
            if self._task_name in TASK_OUTPUTS:
                self._output_schema = get_output_schema(self._task_name)
        else:
            # use base pipeline __call__
            if self._task_name in TASK_INPUTS and self._task_name in TASK_OUTPUTS:
                # delete the first default input which is defined by task.
                self._parameters_schema = generate_pipeline_parameters_schema(
                    parameters)
                self._input_schema = get_input_schema(self._task_name, None)
                self._output_schema = get_output_schema(self._task_name)
            else:
                logger.warning(
                    'Task: %s input is defined: %s, output is defined: %s which is not completed'
                    % (self._task_name, self._task_name
                       in TASK_INPUTS, self._task_name in TASK_OUTPUTS))
                self._input_schema = None
                self._output_schema = None
                if self._task_name in TASK_INPUTS:
                    self._input_schema = get_input_schema(
                        self._task_name, None)
                if self._task_name in TASK_OUTPUTS:
                    self._output_schema = get_output_schema(self._task_name)
                self._parameters_schema = generate_pipeline_parameters_schema(
                    parameters)

    @property
    def task_name(self):
        return self._task_name

    @property
    def is_custom_call(self):
        return self._is_custom_call_method

    @property
    def input_schema(self):
        return self._input_schema

    @property
    def output_schema(self):
        return self._output_schema

    @property
    def parameters_schema(self):
        return self._parameters_schema

    @property
    def schema(self):
        return {
            'input': self._input_schema if self._input_schema else
            self._parameters_schema,  # all parameter is input
            'parameters':
            self._parameters_schema if self._input_schema else {},
            'output': self._output_schema if self._output_schema else {
                'type': 'object',
            },
        }


def is_url(url: str):
    """Check the input url is valid url.

    Args:
        url (str): The url

    Returns:
        bool: If is url return True, otherwise False.
    """
    url_parsed = urlparse(url)
    if url_parsed.scheme in ('http', 'https', 'oss'):
        return True
    else:
        return False


def decode_base64_to_image(content):
    if content.startswith('http') or content.startswith('oss'):
        return content

    from PIL import Image
    image_file_content = base64.b64decode(content)
    return Image.open(BytesIO(image_file_content))


def decode_base64_to_audio(content):
    if content.startswith('http') or content.startswith('oss'):
        return content

    file_content = base64.b64decode(content)
    return file_content


def decode_base64_to_video(content):
    if content.startswith('http') or content.startswith('oss'):
        return content

    file_content = base64.b64decode(content)
    return file_content


def return_origin(content):
    return content


def decode_box(content):
    pass


def service_multipart_input_to_pipeline_input(body):
    """Convert multipart data to pipeline input.

    Args:
        body (dict): The multipart data body
    """
    pass


def pipeline_output_to_service_multipart_output(output):
    """Convert multipart data to service multipart output.

    Args:
        output (dict): Multipart body.
    """
    pass


base64_decoder_map = {
    InputType.IMAGE: decode_base64_to_image,
    InputType.TEXT: return_origin,
    InputType.AUDIO: decode_base64_to_audio,
    InputType.VIDEO: decode_base64_to_video,
    InputType.BOX: decode_box,
    InputType.DICT: return_origin,
    InputType.LIST: return_origin,
    InputType.NUMBER: return_origin,
}


def call_pipeline_with_json(pipeline_info: PipelineInfomation,
                            pipeline: Pipeline, body: str):
    """Call pipeline with json input.

    Args:
        pipeline_info (PipelineInfomation): The pipeline information object.
        pipeline (Pipeline): The pipeline object.
        body (Dict): The input object, include input and parameters
    """
    if pipeline_info.is_custom_call:
        pipeline_inputs = body['input']
        result = pipeline(**pipeline_inputs)
    else:
        pipeline_inputs, parameters = service_base64_input_to_pipeline_input(
            pipeline_info.task_name, body)
        result = pipeline(pipeline_inputs, **parameters)

    return result


def service_base64_input_to_pipeline_input(task_name, body):
    """Convert service base64 input to pipeline input and parameters

    Args:
        task_name (str): The task name.
        body (Dict): The input object, include input and parameters
    """
    if 'input' not in body:
        raise ValueError('No input data!')
    service_input = body['input']
    if 'parameters' in body:
        parameters = body['parameters']
    else:
        parameters = {}
    pipeline_input = {}

    if isinstance(service_input, (str, int, float)):
        return service_input, parameters
    task_input_info = TASK_INPUTS[task_name]
    if isinstance(task_input_info, str):  # no input key default
        if isinstance(service_input, dict):
            return base64_decoder_map[task_input_info](list(
                service_input.values())[0]), parameters
        else:
            return base64_decoder_map[task_input_info](
                service_input), parameters
    elif isinstance(task_input_info, tuple):
        pipeline_input = tuple(service_input)
        return pipeline_input, parameters
    elif isinstance(task_input_info, dict):
        for key, value in service_input.items(
        ):  # task input has no nesting field.
            # get input filed type
            input_type = task_input_info[key]
            # TODO recursion for list, dict if need.
            if not isinstance(input_type, str):
                pipeline_input[key] = value
                continue
            if input_type not in INPUT_TYPE:
                raise ValueError('Invalid input field: %s' % input_type)
            pipeline_input[key] = base64_decoder_map[input_type](value)
        return pipeline_input, parameters
    elif isinstance(task_input_info,
                    list):  # one of input format, we use dict.
        for item in task_input_info:
            if isinstance(item, dict):
                for key, value in service_input.items(
                ):  # task input has no nesting field.
                    # get input filed type
                    input_type = item[key]
                    if input_type not in INPUT_TYPE:
                        raise ValueError('Invalid input field: %s'
                                         % input_type)
                    pipeline_input[key] = base64_decoder_map[input_type](value)
                return pipeline_input, parameters
    else:
        return service_input, parameters


def encode_numpy_image_to_base64(image):
    from PIL import Image
    with BytesIO() as output_bytes:
        pil_image = Image.fromarray(image.astype(np.uint8))
        pil_image.save(output_bytes, 'PNG')
        bytes_data = output_bytes.getvalue()
    base64_str = str(base64.b64encode(bytes_data), 'utf-8')
    return base64_str


def encode_video_to_base64(video):
    return str(base64.b64encode(video), 'utf-8')


def encode_pcm_to_base64(pcm):
    return str(base64.b64encode(pcm), 'utf-8')


def encode_wav_to_base64(wav):
    return str(base64.b64encode(wav), 'utf-8')


def encode_bytes_to_base64(bts):
    return str(base64.b64encode(bts), 'utf-8')


base64_encoder_map = {
    'image': encode_numpy_image_to_base64,
    'video': encode_video_to_base64,
    'pcm': encode_pcm_to_base64,
    'wav': encode_wav_to_base64,
    'bytes': encode_bytes_to_base64,
}

# convert numpy etc type to python type.
type_to_python_type = {
    np.int64: int,
}


def _convert_to_python_type(inputs):
    if isinstance(inputs, (list, tuple)):
        res = []
        for item in inputs:
            res.append(_convert_to_python_type(item))
        return res
    elif isinstance(inputs, dict):
        res = {}
        for k, v in inputs.items():
            if type(v) in type_to_python_type:
                res[k] = type_to_python_type[type(v)](v)
            else:
                res[k] = _convert_to_python_type(v)
        return res
    elif isinstance(inputs, np.ndarray):
        return inputs.tolist()
    else:
        return inputs


def pipeline_output_to_service_base64_output(task_name, pipeline_output):
    """Convert pipeline output to service output,
    convert binary fields to base64 encoding。

    Args:
        task_name (str): The output task name.
        pipeline_output (object): The pipeline output.
    """
    json_serializable_output = {}
    task_outputs = []
    if task_name in TASK_OUTPUTS:
        task_outputs = TASK_OUTPUTS[task_name]
    for key, value in pipeline_output.items():
        if key not in task_outputs:
            continue  # skip the output not defined.
        if key in [
                OutputKeys.OUTPUT_IMG, OutputKeys.OUTPUT_IMGS,
                OutputKeys.OUTPUT_VIDEO, OutputKeys.OUTPUT_PCM,
                OutputKeys.OUTPUT_PCM_LIST, OutputKeys.OUTPUT_WAV
        ]:
            if isinstance(value, list):
                items = []
                if key == OutputKeys.OUTPUT_IMGS:
                    output_item_type = OutputKeys.OUTPUT_IMG
                else:
                    output_item_type = OutputKeys.OUTPUT_PCM
                for item in value:
                    items.append(base64_encoder_map[
                        OutputTypes[output_item_type]](item))
                json_serializable_output[key] = items
            else:
                json_serializable_output[key] = base64_encoder_map[
                    OutputTypes[key]](
                        value)
        elif OutputTypes[key] in [np.ndarray] and isinstance(
                value, np.ndarray):
            json_serializable_output[key] = value.tolist()
        elif isinstance(value, np.ndarray):
            json_serializable_output[key] = value.tolist()
        else:
            json_serializable_output[key] = value

    return _convert_to_python_type(json_serializable_output)
