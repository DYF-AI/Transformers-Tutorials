# 参考paddle的推理过程
# ser(xfun_data) + re 数据构造
# @date: 20221227

import os
import json

from utils.image_utils import *

from transformers import AutoTokenizer
from modeling_layoutxlm_re import LayoutLMv2ForRelationExtraction

DP = "/mnt/j/dataset/document-intelligence/XFUND/fr"
OP = "/mnt/j/dataset/document-intelligence/XFUND/output"
MP = "/mnt/j/model/pretrained-model/bert_torch"

TRAINED_MP = "./output/layoutxlm-finetuned-xfund-re/"

use_visual_backbone = False

# tokenizer = AutoTokenizer.from_pretrained(os.path.join(TRAINED_MP, "checkpoint-5000"))
tokenizer = AutoTokenizer.from_pretrained(os.path.join(MP, "xlm-roberta-base"))

def make_ser_input(image_xfund_data):
    ser_inputs = []
    size = [json_data["img"]["width"], json_data["img"]["height"]]
    # 现在读取整张图的信息，然后再安找512长度进行切分
    tokenized_doc = {"input_ids": [], "bbox": [], "labels": []}
    # 实体、关系、实体id及label, 实体id对应index
    entities, relations, id2label, entity_id_to_index_map = [], [], {}, {}
    entities_all_message = []
    entity_idx_dict = {}
    empty_entity = set()
    idx = 0
    for line in json_data['document']:
        print(line)
        if len(line['text']) == 0:
            empty_entity.add(line['id'])
            continue
        tokenized_inputs = tokenizer(line["text"],
                                     add_special_tokens=False,
                                     return_offsets_mapping=True,
                                     return_attention_mask=False,
                                     )
        # bbox是切词后词的box
        text_length, ocr_length, bbox = 0, 0, []
        # 计算每个词的box
        for token_id, offset in zip(tokenized_inputs["input_ids"], tokenized_inputs["offset_mapping"]):
            if token_id == 6:
                bbox.append(None)
                continue
            text_length += offset[1] - offset[0]
            tmp_box = []
            # 这里统一json的切词和tokenizer的切词
            while ocr_length < text_length:
                ocr_word = line["words"].pop(0)
                ocr_length += len(
                    tokenizer._tokenizer.normalizer.normalize_str(ocr_word["text"].strip())
                )
                tmp_box.append(simplify_bbox(ocr_word["box"]))
            if len(tmp_box) == 0:
                tmp_box = last_box
            bbox.append(normalize_bbox(merge_bbox(tmp_box), size))
            last_box = tmp_box  # noqa
        # 当box为None时,替换
        bbox = [
            [bbox[i + 1][0], bbox[i + 1][1], bbox[i + 1][0], bbox[i + 1][1]] if b is None else b
            for i, b in enumerate(bbox)
        ]
        # 生成每个字或者词的标签
        if line["label"] == "other":
            label = ["O"] * len(bbox)
        else:
            label = [f"I-{line['label'].upper()}"] * len(bbox)
            label[0] = f"B-{line['label'].upper()}"
        tokenized_inputs.update({"bbox": bbox, "labels": label})
        if label[0] != "O":
            entity_id_to_index_map[line["id"]] = len(entities)
            entity_idx_dict[len(entities)] = line["id"]
            entities.append(
                {
                    "start": len(tokenized_doc["input_ids"]),
                    "end": len(tokenized_doc["input_ids"]) + len(tokenized_inputs["input_ids"]),
                    "label": line["label"].upper(),
                    "id": line["id"]
                }
            )
            # 记录entitytext、bbox信息
            entities_all_message.append(
                {
                    "transcription": line['text'],
                    "bbox": line["bbox"],
                    "pred_id": None,
                    "pred": line["label"],
                    "id": line["id"]
                }
            )
        for i in tokenized_doc:
            tokenized_doc[i] = tokenized_doc[i] + tokenized_inputs[i]
    # 输入词的长度是512
    chunk_size = 512
    for chunk_id, index in enumerate(range(0, len(tokenized_doc["input_ids"]), chunk_size)):
        item = {}
        # k: input_ids, bbox, labels
        for k in tokenized_doc:
            item[k] = tokenized_doc[k][index: index + chunk_size]
        entities_in_this_span = []
        entities_all_message_in_this_span = []
        global_to_local_map = {}
        for entity_id, entity in enumerate(entities):
            # 确保实体是512以内
            if (
                    index <= entity["start"] < index + chunk_size
                    and index <= entity["end"] < index + chunk_size
            ):
                entity["start"] = entity["start"] - index
                entity["end"] = entity["end"] - index
                global_to_local_map[entity_id] = len(entities_in_this_span)
                entities_in_this_span.append(entity)
        item.update(
            {
                "id": f"{json_data['id']}_{chunk_id}",
                "entities": entities_in_this_span,
            }
        )
        ser_inputs.append(item)
        print("item:", item)

# 根据xfund的数据，构造ser的输入输出
# re模型的输入：ser的输入+ser的输出
# 暂时只管一张图的输入吧
def make_ser_result(image_xfund_data):
    ser_results = []
    empty_entity = set()
    # 遍历每个json的每个box(实体)
    for line in json_data['document']:
        if len(line['text']) == 0:
            empty_entity.add(line['id'])
            continue
        # 模拟ser生成的结果
        ser_result = {
            "transcription": line['text'],
            "bbox": line["bbox"],
            "pred_id": None,
            "pred": line["label"]
        }
        ser_results.append(ser_result)
    return ser_results


def make_input(ser_inputs, ser_results):
    pass



class RelationPredictor(object):
    def __init__(self, use_visual_backbone=False):
        self.use_visual_backbone = use_visual_backbone

    def __call__(self, xfund_data):
       pass



if __name__ == "__main__":
    input_json = "./ser_entities_sample_zh.json"
    with open(input_json, 'r', encoding='utf-8') as fi:
        json_data = json.load(fi)
        print(json_data)

    # 创建re的对象
    re = RelationPredictor()
