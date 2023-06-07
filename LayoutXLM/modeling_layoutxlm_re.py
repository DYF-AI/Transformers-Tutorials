
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from transformers import LayoutLMv2PreTrainedModel, requires_backends
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, TokenClassifierOutput
from transformers.models.layoutlmv2.modeling_layoutlmv2 import LayoutLMv2Embeddings, LayoutLMv2VisualBackbone, \
    LayoutLMv2Encoder, LayoutLMv2Pooler, LAYOUTLMV2_INPUTS_DOCSTRING, _CONFIG_FOR_DOC
from transformers.pytorch_utils import torch_int_div
from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings

import copy
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers.utils.generic import ModelOutput


@dataclass
class RelationExtractionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    pred_relations: dict = None
    entities: dict = None
    relations: dict = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# 关系抽取解码
class LayoutLMv2RelationExtractionDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 实体的embedding
        self.entity_emb = nn.Embedding(3, config.hidden_size, scale_grad_by_freq=True)
        projection = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
        )
        self.ffnn_head = copy.deepcopy(projection)
        self.ffnn_tail = copy.deepcopy(projection)
        self.rel_classifier = BiAffineAttention(config.hidden_size // 2, 2)
        self.loss_fct = CrossEntropyLoss()

    def build_relation(self, relations, entities):
        batch_size = len(relations)
        new_relations = []
        for b in range(batch_size):
            if len(entities[b]["start"]) <= 2:
                entities[b] = {"end": [1, 1], "label": [0, 0], "start": [0, 0]}
            # 两层循环, 所有实体都存在连接的可能
            all_possible_relations = set(
                [
                    (i, j)
                    for i in range(len(entities[b]["label"]))
                    for j in range(len(entities[b]["label"]))
                    if entities[b]["label"][i] == 1 and entities[b]["label"][j] == 2
                ]
            )
            if len(all_possible_relations) == 0:
                all_possible_relations = set([(0, 1)])
            positive_relations = set(list(zip(relations[b]["head"], relations[b]["tail"])))
            negative_relations = all_possible_relations - positive_relations
            positive_relations = set([i for i in positive_relations if i in all_possible_relations])
            reordered_relations = list(positive_relations) + list(negative_relations)

            relation_per_doc = {"head": [], "tail": [], "label": []}
            relation_per_doc["head"] = [i[0] for i in reordered_relations]
            relation_per_doc["tail"] = [i[1] for i in reordered_relations]
            relation_per_doc["label"] = [1] * len(positive_relations) \
                                        + [0] * (len(reordered_relations) - len(positive_relations))
            assert len(relation_per_doc["head"]) != 0
            new_relations.append(relation_per_doc)
        return new_relations, entities

    def get_predicted_relations(self, logits, relations, entities):
        pred_relations = []
        for i, pred_label in enumerate(logits.argmax(-1)):
            if pred_label != 1:
                continue
            rel = {}
            rel["head_id"] = relations["head"][i]
            rel["head"] = (entities["start"][rel["head_id"]], entities["end"][rel["head_id"]])
            rel["head_type"] = entities["label"][rel["head_id"]]

            rel["tail_id"] = relations["tail"][i]
            rel["tail"] = (entities["start"][rel["tail_id"]], entities["end"][rel["tail_id"]])
            rel["tail_type"] = entities["label"][rel["tail_id"]]
            rel["type"] = 1
            pred_relations.append(rel)
        return pred_relations

    # 在forwar中尽量使用tensor计算，非tensor信息以入参形式输入
    def forward(self, hidden_states, entities, relations):
        batch_size, max_n_words, context_dim = hidden_states.size()
        device = hidden_states.device
        # TODO: 移出forwar
        relations, entities = self.build_relation(relations, entities)
        loss = 0
        all_pred_relations = []
        for b in range(batch_size):
            head_entities = torch.tensor(relations[b]["head"], device=device)
            tail_entities = torch.tensor(relations[b]["tail"], device=device)
            relation_labels = torch.tensor(relations[b]["label"], device=device)
            entities_start_index = torch.tensor(entities[b]["start"], device=device)
            entities_labels = torch.tensor(entities[b]["label"], device=device)
            head_index = entities_start_index[head_entities]
            head_label = entities_labels[head_entities]
            head_label_repr = self.entity_emb(head_label)

            tail_index = entities_start_index[tail_entities]
            tail_label = entities_labels[tail_entities]
            tail_label_repr = self.entity_emb(tail_label)

            head_repr = torch.cat(
                (hidden_states[b][head_index], head_label_repr),
                dim=-1,
            )
            tail_repr = torch.cat(
                (hidden_states[b][tail_index], tail_label_repr),
                dim=-1,
            )
            heads = self.ffnn_head(head_repr)
            tails = self.ffnn_tail(tail_repr)
            # 这个只能一个batch一个batch地预测吗？
            logits = self.rel_classifier(heads, tails)
            loss += self.loss_fct(logits, relation_labels)
            pred_relations = self.get_predicted_relations(logits, relations[b], entities[b])
            all_pred_relations.append(pred_relations)
        return loss, all_pred_relations

class ViLayoutLMv2Model(LayoutLMv2PreTrainedModel):
    def __init__(self, config, use_visual_backbone=False):
        #requires_backends(self, "detectron2")
        super().__init__(config)
        self.config = config
        self.use_visual_backbone = use_visual_backbone
        print("use_visual_backbone:", use_visual_backbone)
        self.has_visual_segment_embedding = config.has_visual_segment_embedding
        self.embeddings = LayoutLMv2Embeddings(config)

        # change 1:
        if self.use_visual_backbone is True:
            self.visual = LayoutLMv2VisualBackbone(config)
            self.visual_proj = nn.Linear(config.image_feature_pool_shape[-1], config.hidden_size)

        if self.has_visual_segment_embedding:
            self.visual_segment_embedding = nn.Parameter(nn.Embedding(1, config.hidden_size).weight[0])
        self.visual_LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.visual_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.encoder = LayoutLMv2Encoder(config)
        self.pooler = LayoutLMv2Pooler(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _calc_text_embeddings(self, input_ids, bbox, position_ids, token_type_ids, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        if inputs_embeds is None:
            inputs_embeds = self.embeddings.word_embeddings(input_ids)
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        spatial_position_embeddings = self.embeddings._calc_spatial_position_embeddings(bbox)
        token_type_embeddings = self.embeddings.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + spatial_position_embeddings + token_type_embeddings
        embeddings = self.embeddings.LayerNorm(embeddings)
        embeddings = self.embeddings.dropout(embeddings)
        return embeddings

    def _calc_img_embeddings(self, image, bbox, position_ids):
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        spatial_position_embeddings = self.embeddings._calc_spatial_position_embeddings(bbox)

        # change 2
        if self.use_visual_backbone is True:
            visual_embeddings = self.visual_proj(self.visual(image))
            embeddings = visual_embeddings + position_embeddings + spatial_position_embeddings
        else:
            embeddings =  position_embeddings + spatial_position_embeddings

        if self.has_visual_segment_embedding:
            embeddings += self.visual_segment_embedding
        embeddings = self.visual_LayerNorm(embeddings)
        embeddings = self.visual_dropout(embeddings)
        return embeddings

    def _calc_visual_bbox(self, image_feature_pool_shape, bbox, device, final_shape):
        visual_bbox_x = torch_int_div(
            torch.arange(
                0,
                1000 * (image_feature_pool_shape[1] + 1),
                1000,
                device=device,
                dtype=bbox.dtype,
            ),
            self.config.image_feature_pool_shape[1],
        )
        visual_bbox_y = torch_int_div(
            torch.arange(
                0,
                1000 * (self.config.image_feature_pool_shape[0] + 1),
                1000,
                device=device,
                dtype=bbox.dtype,
            ),
            self.config.image_feature_pool_shape[0],
        )
        visual_bbox = torch.stack(
            [
                visual_bbox_x[:-1].repeat(image_feature_pool_shape[0], 1),
                visual_bbox_y[:-1].repeat(image_feature_pool_shape[1], 1).transpose(0, 1),
                visual_bbox_x[1:].repeat(image_feature_pool_shape[0], 1),
                visual_bbox_y[1:].repeat(image_feature_pool_shape[1], 1).transpose(0, 1),
            ],
            dim=-1,
        ).view(-1, bbox.size(-1))

        visual_bbox = visual_bbox.repeat(final_shape[0], 1, 1)

        return visual_bbox

    #@add_start_docstrings_to_model_forward(LAYOUTLMV2_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    #@replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        image: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Return:
        Examples:
        ```python
        >>> from transformers-1 import LayoutLMv2Processor, LayoutLMv2Model, set_seed
        >>> from PIL import Image
        >>> import torch
        >>> from datasets import load_dataset
        >>> set_seed(88)
        >>> processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")
        >>> model = ViLayoutLMv2Model.from_pretrained("microsoft/layoutlmv2-base-uncased")
        >>> dataset = load_dataset("hf-internal-testing/fixtures_docvqa")
        >>> image_path = dataset["test"][0]["file"]
        >>> image = Image.open(image_path).convert("RGB")
        >>> encoding = processor(image, return_tensors="pt")
        >>> outputs = model(**encoding)
        >>> last_hidden_states = outputs.last_hidden_state
        >>> last_hidden_states.shape
        torch.Size([1, 342, 768])
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        visual_shape = list(input_shape)
        visual_shape[1] = self.config.image_feature_pool_shape[0] * self.config.image_feature_pool_shape[1]
        visual_shape = torch.Size(visual_shape)
        final_shape = list(input_shape)
        final_shape[1] += visual_shape[1]
        final_shape = torch.Size(final_shape)

        visual_bbox = self._calc_visual_bbox(self.config.image_feature_pool_shape, bbox, device, final_shape)
        final_bbox = torch.cat([bbox, visual_bbox], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        # change 3:
        if self.use_visual_backbone is True:
            visual_attention_mask = torch.ones(visual_shape, device=device)
        else:
            visual_attention_mask = torch.zeros(visual_shape, device=device)

        final_attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if position_ids is None:
            seq_length = input_shape[1]
            position_ids = self.embeddings.position_ids[:, :seq_length]
            position_ids = position_ids.expand(input_shape)

        visual_position_ids = torch.arange(0, visual_shape[1], dtype=torch.long, device=device).repeat(
            input_shape[0], 1
        )
        final_position_ids = torch.cat([position_ids, visual_position_ids], dim=1)

        if bbox is None:
            bbox = torch.zeros(tuple(list(input_shape) + [4]), dtype=torch.long, device=device)

        text_layout_emb = self._calc_text_embeddings(
            input_ids=input_ids,
            bbox=bbox,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        visual_emb = self._calc_img_embeddings(
            image=image,
            bbox=visual_bbox,
            position_ids=visual_position_ids,
        )
        final_emb = torch.cat([text_layout_emb, visual_emb], dim=1)

        extended_attention_mask = final_attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            final_emb,
            extended_attention_mask,
            bbox=final_bbox,
            position_ids=final_position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class LayoutLMv2ForRelationExtraction(LayoutLMv2PreTrainedModel):
    def __init__(self, config, use_visual_backbone=False):
        super().__init__(config)

        #self.layoutlmv2 = LayoutLMv2Model(config)
        self.layoutlmv2 = ViLayoutLMv2Model(config,use_visual_backbone=use_visual_backbone)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.extractor = LayoutLMv2RelationExtractionDecoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    #@add_start_docstrings_to_model_forward(LAYOUTLMV2_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    #@replace_return_docstrings(output_type=RelationExtractionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        image: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        entities: Optional[dict] = None,
        relations: Optional[dict] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        entities (`list[dict]`):
            List of dictionaries (one per example in the batch). Each dictionary contains 3 keys: `start`, `end` and
            `label`.
        relations (`list[dict]`):
            List of dictionaries (one per example in the batch). Each dictionary contains 4 keys: `start_index`,
            `end_index`, `head` and `tail`.

        Returns:

        Example:

        ```
        python
        >>> from transformers-1 import LayoutLMv2Processor,#LayoutLMv2ForRelationExtraction
        >>> from PIL import Image
        >>> from datasets import load_dataset

        >>> processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")
        >>> model = LayoutLMv2ForRelationExtraction.from_pretrained("microsoft/layoutlmv2-base-uncased")

        >>> dataset = load_dataset("hf-internal-testing/fixtures_docvqa")
        >>> image_path = dataset["test"][0]["file"]
        >>> image = Image.open(image_path).convert("RGB")
        >>> encoding = processor(image, return_tensors="pt")

        >>> # instantiate relations as empty at inference time
        >>> encoding["entities"] = [{"start": [0, 4], "end": [3, 6], "label": [2, 1]}]
        >>> encoding["relations"] = [{"start_index": [], "end_index": [], "head": [], "tail": []}]

        >>> outputs = model(**encoding)
        >>> predicted_relations = outputs.pred_relations[0]
        ```
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlmv2(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        seq_length = input_ids.size(1) if input_ids is not None else inputs_embeds.size(1)
        text_output = outputs[0][:, :seq_length]
        text_output = self.dropout(text_output)  # (batchsize,512,768)

        if entities is None or relations is None:
            raise ValueError(
                "You need to provide entities and relations. Instantiate relations with empty lists at inference time"
            )
        loss, pred_relations = self.extractor(text_output, entities, relations)

        if not return_dict:
            output = (loss, pred_relations, entities, relations) + outputs[2:]
            return output

        return RelationExtractionOutput(
            loss=loss,
            pred_relations=pred_relations,
            entities=entities,
            relations=relations,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


from transformers import LayoutLMv2FeatureExtractor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


# 双反射注意力机制
class BiAffineAttention(nn.Module):
    def __init__(self, in_features, out_feature):
        super(BiAffineAttention, self).__init__()

        self.in_feature = in_features
        self.out_feature = out_feature

        self.bilinear = nn.Bilinear(in_features, in_features, out_feature, bias=False)
        self.linear = nn.Linear(2 * in_features, out_feature, bias=True)

        self.reset_parameters()

    def forward(self, x_1, x_2):
        return self.bilinear(x_1, x_2) + self.linear(torch.cat((x_1, x_2), dim=-1))

    def reset_parameters(self):
        self.bilinear.reset_parameters()
        self.linear.reset_parameters()


