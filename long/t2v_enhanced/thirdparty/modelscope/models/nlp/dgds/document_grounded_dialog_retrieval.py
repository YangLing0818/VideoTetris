# These codes are copied from modelscope revision c58451baead80d83281f063d12fb377fad415257 
import os
from typing import Dict

import torch

from modelscope.metainfo import Models
from modelscope.models.base import Tensor, TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from .backbone import DPRModel


@MODELS.register_module(
    Tasks.document_grounded_dialog_retrieval, module_name=Models.doc2bot)
class DocumentGroundedDialogRetrievalModel(TorchModel):
    _backbone_prefix = ''

    def __init__(self, model_dir, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.config = Config.from_file(
            os.path.join(self.model_dir, ModelFile.CONFIGURATION))
        self.model = DPRModel(model_dir, self.config)
        state_dict = torch.load(
            os.path.join(self.model_dir, ModelFile.TORCH_MODEL_BIN_FILE),
            map_location='cpu')
        self.model.load_state_dict(state_dict)

    def forward(self, input: Dict[str, Tensor], gck_segment=32):
        query_input_ids = input['query_input_ids']
        query_attention_mask = input['query_attention_mask']
        context_input_ids = input['context_input_ids']
        context_attention_mask = input['context_attention_mask']
        labels = input['labels']
        outputs = self.model(query_input_ids, query_attention_mask,
                             context_input_ids, context_attention_mask, labels,
                             gck_segment)
        return outputs

    def encode_query(self, input: Dict[str, Tensor]):
        query_input_ids = input['query_input_ids']
        query_attention_mask = input['query_attention_mask']
        query_vector = self.model.qry_encoder(query_input_ids,
                                              query_attention_mask, None)
        return query_vector

    def encode_context(self, input: Dict[str, Tensor]):
        context_input_ids = input['context_input_ids']
        context_attention_mask = input['context_attention_mask']
        context_vector = self.model.ctx_encoder(context_input_ids,
                                                context_attention_mask, None)
        return context_vector
