import os.path

import torch
from transformers import  AutoTokenizer, AutoConfig
from transformers import LayoutLMv2Model,LayoutLMv2ForTokenClassification
from modeling_layoutxlm_re import LayoutLMv2ForRelationExtraction

TRAINED_MP = "./output/layoutxlm-finetuned-xfund-re/"
MP = r"J:\model\pretrained-model\bert_torch"
# re训练的模型
use_visual_backbone = False
model = LayoutLMv2ForRelationExtraction.from_pretrained(
    os.path.join(TRAINED_MP, "checkpoint-5000"), use_visual_backbone=use_visual_backbone)

model_layoutxlm = model.layoutlmv2

model_input = {'input_ids': torch.zeros((1, 512), dtype=torch.int64) + 10,
               'bbox': torch.zeros((1, 512, 4), dtype=torch.int64) + 20,
               'images': torch.zeros((3, 224, 224), dtype=torch.float32),
               'attention_mask': torch.zeros((1, 512), dtype=torch.int64) + 1}

torch.onnx.export(
    model_layoutxlm,
    #tuple((model_input['input_ids'], model_input['bbox'], model_input['images'], model_input['attention_mask'])),
    tuple((model_input['input_ids'], model_input['bbox'],None, model_input['attention_mask'])),
    f="v2.onnx",
    input_names=['input_ids', 'bbox', 'images', 'attention_mask'],
    output_names=['logits'],
    dynamic_axes={'input_ids': {0: 'batch_size'},
                  'bbox': {0: 'batch_size'},
                  'attention_mask': {0: 'batch_size'},
                  },
    do_constant_folding=True,
    opset_version=13,

)