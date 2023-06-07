#!/usr/bin/env python
# coding: utf-8

# In[12]:


import copy
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import datasets
from datasets import load_dataset,load_from_disk
from datasets import Dataset, DatasetDict

from transformers import AutoTokenizer
from modeling_layoutxlm_re import LayoutLMv2ForRelationExtraction


# In[13]:


import os
DP = "/mnt/j/dataset/document-intelligence/XFUND/fr"
OP = "/mnt/j/dataset/document-intelligence/XFUND/output"
MP = "/mnt/j/model/pretrained-model/torch"

use_visual_backbone=False

# tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutxlm-base")
tokenizer = AutoTokenizer.from_pretrained(os.path.join(MP, "microsoft-layoutxlm-base"))


# model = LayoutLMv2ForRelationExtraction.from_pretrained("microsoft/layoutxlm-base")
model = LayoutLMv2ForRelationExtraction.from_pretrained(os.path.join(MP, "microsoft-layoutxlm-base"),use_visual_backbone=use_visual_backbone)


# In[14]:


from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy
from torch.utils.data import DataLoader
from dataclasses import dataclass
import torch

from typing import Optional, Union

from transformers import LayoutLMv2FeatureExtractor
feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)

@dataclass
class DataCollatorForKeyValueExtraction:
    feature_extractor: LayoutLMv2FeatureExtractor
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    use_visual_backbone: bool = False

    def __call__(self, features):
        # prepare image input
        image = self.feature_extractor([feature["original_image"] for feature in features], return_tensors="pt").pixel_values

        # prepare text input
        entities = []
        relations = []
        for feature in features:
            del feature["image"]
            del feature["id"]
            del feature["labels"]
            del feature["original_image"]
            entities.append(feature["entities"])
            del feature["entities"]
            relations.append(feature["relations"])
            del feature["relations"]
      
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )
        if use_visual_backbone:
            batch["image"] = image
        batch["entities"] = entities
        batch["relations"] = relations
        return batch

data_collator = DataCollatorForKeyValueExtraction(
    feature_extractor,
    tokenizer,
    pad_to_multiple_of=8,
    padding="max_length",
    max_length=512,
    use_visual_backbone=use_visual_backbone
)


# In[15]:


train_dataset = Dataset.from_file(os.path.join(DP, "fr.train.arrow"))
test_dataset = Dataset.from_file(os.path.join(DP, "fr.val.arrow"))
dataset = DatasetDict({"train": train_dataset, "validation": test_dataset})
print("dataset:", dataset)


# In[17]:


from utils.evaluation import re_score


def compute_metrics(p):
    pred_relations, gt_relations = p
    score = re_score(pred_relations, gt_relations, mode="boundaries")
    return score


# In[18]:


from transformers import TrainingArguments
from utils.xfun_trainer import XfunReTrainer

lang = "fr"
if use_visual_backbone:
    output_dir = "./output/layoutxlm-finetuned-xfund-re-{}".format(lang)
else:
    output_dir = "./output/vi-layoutxlm-finetuned-xfund-re-{}".format(lang)

training_args = TrainingArguments(output_dir=output_dir,
                                  overwrite_output_dir=True,
                                  remove_unused_columns=False,
                                  max_steps=5000,
                                  per_device_train_batch_size=2,
                                  per_device_eval_batch_size=2,
                                  warmup_ratio=0.1,
                                  learning_rate=7e-5,
                                  #fp16=True,
                                  push_to_hub=False,
                                  do_eval = True,
                                  evaluation_strategy = "steps",
                                  eval_steps = 100,
                                  report_to = "tensorboard",
                                  )

trainer = XfunReTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# In[19]:


trainer.train()


# In[ ]:




